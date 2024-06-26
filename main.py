import argparse
import logging
import multiprocessing
import os
import subprocess
from contextlib import chdir
from datetime import date, datetime, timedelta
from functools import partial
from textwrap import dedent
from typing import NamedTuple

from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_community.llms.gpt4all import GPT4All
from rich.console import Console

MAX_TOKENS = 1024

MAP_PROMPT_TEMPLATE = """
    [INST]
    Generate a concise task description in past tense based on the git commit message and git diff provided.
    Your response should state clearly in a single line what improvements or fixes were made to the code and what task has been achieved.
    The git diff provides all changes made to the code, therefore you can use it to interpret how the code changed.
    {context}
    [/INST]
"""

COLLAPSE_PROMPT_TEMPLATE = """
    [INST]
    Summarise the following documents:
    {doc_summaries}
    [/INST]
"""

REDUCE_PROMPT_TEMPLATE = """
    [INST]
    The following is set of git branches and description of tasks I've done:
    {doc_summaries}
    Take these and distill it into a final, consolidated Markdown list of tasks and achievements.
    [/INST]
    Helpful Answer:
"""

# TODO include branch
CTX_TEMPLATE = """
    Git message:
    {message}

    Branch:
    {branch}

    Git diff:
    {diff}
"""


class Commit(NamedTuple):
    branch: str
    message: str
    diff: str


def load_model():
    local_path = (
        f"{os.environ['HOME']}/.cache/gpt4all/mistral-7b-instruct-v0.1.Q4_0.gguf"
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    llm = GPT4All(
        model=local_path,
        allow_download=True,
        verbose=True,
        n_threads=multiprocessing.cpu_count() - 1,
        streaming=True,
    )
    return llm, ""


def format_document(
    commit: Commit, prompt: PromptTemplate, llm, max_tokens: int = MAX_TOKENS
):
    prompt_value = prompt.format(
        message=commit.message, branch=commit.branch, diff=commit.diff
    )
    n_tokens = llm.get_num_tokens(prompt_value)

    diff = commit.diff
    diff_lines = diff.splitlines()
    if len(diff_lines) == 1 and n_tokens > max_tokens:
        diff = ""  # FIXME better handling

    while n_tokens > max_tokens:
        if not diff_lines:
            break
        diff_lines = diff_lines[:-1]
        diff = "\n".join(diff_lines)
        prompt_value = prompt.format(
            message=commit.message, branch=commit.branch, diff=diff
        )
        n_tokens = llm.get_num_tokens(prompt_value)

    message = commit.message
    while n_tokens > max_tokens:
        message = message[:-2]
        prompt_value = prompt.format(message=message, branch=commit.branch, diff=diff)
        n_tokens = llm.get_num_tokens(prompt_value)

    return prompt_value


def format_docs(docs) -> str:
    return "\n\n".join(
        f"Branch: {doc.metadata['branch']}. Task: {doc.page_content}" for doc in docs
    )


def get_chain():
    llm, system_prompt = load_model()

    document_prompt = PromptTemplate.from_template(system_prompt + dedent(CTX_TEMPLATE))
    map_prompt = PromptTemplate.from_template(
        system_prompt + dedent(MAP_PROMPT_TEMPLATE)
    )

    map_prompt_tokens = llm.get_num_tokens(map_prompt.format(context=""))
    partial_format_document = partial(
        format_document,
        prompt=document_prompt,
        llm=llm,
        max_tokens=MAX_TOKENS - map_prompt_tokens,
    )

    map_chain = (
        {"context": partial_format_document} | map_prompt | llm | StrOutputParser()
    ).with_config(run_name="Summarize commit")
    map_as_doc_chain = RunnableParallel(
        {"doc": RunnablePassthrough(), "content": map_chain}
    ) | (
        lambda x: Document(
            page_content=x["content"], metadata={"branch": x["doc"].branch}
        )
    )

    collapse_chain = (
        {"doc_summaries": format_docs}
        | PromptTemplate.from_template(system_prompt + dedent(COLLAPSE_PROMPT_TEMPLATE))
        | llm
        | StrOutputParser()
    )

    def get_num_tokens(docs):
        return llm.get_num_tokens(format_docs(docs))

    def collapse(docs, config, max_tokens=MAX_TOKENS):
        collapse_ct = 1
        while get_num_tokens(docs) > max_tokens:
            config["run_name"] = f"Collapse {collapse_ct}"
            invoke = partial(collapse_chain.invoke, config=config)
            split_docs = split_list_of_docs(docs, get_num_tokens, max_tokens)
            docs = [collapse_docs(_docs, invoke) for _docs in split_docs]
            collapse_ct += 1
        return docs

    reduce_chain = (
        {"doc_summaries": format_docs}
        | PromptTemplate.from_template(dedent(REDUCE_PROMPT_TEMPLATE))
        | llm
        | StrOutputParser()
    ).with_config(run_name="Reduce")

    map_reduce = (map_as_doc_chain.map() | collapse | reduce_chain).with_config(
        run_name="Map Reduce"
    )

    def run_llm(commits: list[Commit]) -> str:
        return map_reduce.invoke(commits, config={"max_concurrency": 1})

    return run_llm


def _parse_git_log(branch: str, output: bytes, seen_commits: set[str]) -> list[Commit]:
    commits: list[Commit] = []
    out_lines = output.decode("utf-8").split("\n")

    for line in out_lines:
        if not line:
            continue

        commit_hash, message = line.split(" ", maxsplit=1)

        if commit_hash in seen_commits:
            continue

        sp = subprocess.run(
            ["git", "--no-pager", "diff", "--minimal", f"{commit_hash}^!"],
            capture_output=True,
        )

        if sp.returncode != 0:
            raise RuntimeError(
                f"git diff command returned non-zero status {sp.returncode}\n{sp.stderr.decode('utf-8')}"
            )

        seen_commits.add(commit_hash)
        diff = sp.stdout.decode("utf-8")
        commits.append(Commit(branch, message, diff))

    return commits


def get_commits(
    repo_path: str, since_date: date, author_name: str | None
) -> list[Commit]:
    since_str = since_date.isoformat()
    cmd = [
        "git",
        "--no-pager",
        "log",
        "--no-decorate",
        "--oneline",
        "--no-merges",
        "--all",
        "--since",
        since_str,
    ]
    if author_name is not None:
        cmd.append(f"--author={author_name}")

    commits: list[Commit] = []

    with chdir(repo_path):
        sp = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"], capture_output=True
        )

        if sp.returncode != 0:
            raise RuntimeError(
                f"git branch command exited with non-zero status {sp.returncode}\n{sp.stderr.decode('utf-8')}"
            )

        branches = sp.stdout.decode("utf-8").splitlines()
        seen_commits: set[str] = set()

        for branch in branches:
            # NOTE limit to prevent errors due to ill-named branches
            branch = branch[:75]
            branch_cmd = [*cmd, f"--branches=*{branch}"]
            sp = subprocess.run(branch_cmd, capture_output=True)

            if sp.returncode != 0:
                raise RuntimeError(
                    f"git log command returned non-zero status {sp.returncode}\n{sp.stderr.decode('utf-8')}"
                )

            commits.extend(_parse_git_log(branch, sp.stdout, seen_commits))

    return commits


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--repo-paths", type=str, nargs="+", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="limit to the number of commits to analyze in each repository",
    )
    parser.add_argument("--author", type=str, help="author name to filter commits")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--lookback",
        type=int,
        default=1,
        help="number of days to look back in git logs",
    )
    args = parser.parse_args()

    # FIXME something prettier to hide initial logging
    logging.disable(logging.CRITICAL)

    import transformers

    logging.disable(logging.NOTSET)

    transformers.logging.set_verbosity_error()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)

    if args.debug:
        set_debug(True)
        transformers.logging.set_verbosity_debug()
        logger.setLevel(logging.DEBUG)

    console = Console()
    since_date = (datetime.now() - timedelta(days=args.lookback)).date()

    repo_outputs = []
    for repo_path in args.repo_paths:
        commits = get_commits(repo_path, since_date, args.author)

        if not commits:
            console.log(
                f"[bold red]WARNING[/bold red]: no commits in {repo_path} since {since_date}"
            )
            continue

        if len(commits) >= args.limit:
            console.log(f"limiting from {len(commits)} to {args.limit} commits")
            commits = commits[: args.limit]

        with console.status(
            f"[bold green]Summarizing tasks from {len(commits)} commits in {repo_path}..."
        ):
            chain = get_chain()
            llm_output = chain(commits)
            repo_outputs.append((repo_path, llm_output))

    print()
    for path, output in sorted(repo_outputs, key=lambda x: x[0]):
        console.print(f"[bold green] • Summary for repository {path}")
        console.print(output)
        print()


if __name__ == "__main__":
    main()
