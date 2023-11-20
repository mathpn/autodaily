import argparse
import os
import subprocess
from contextlib import chdir
from textwrap import dedent
from typing import NamedTuple

from langchain.chains import (
    LLMChain,
    MapReduceDocumentsChain,
    ReduceDocumentsChain,
    StuffDocumentsChain,
)
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.schema import Document

MAP_PROMPT_TEMPLATE = """
    [INST]
    Generate a concise task description in past tense based on the git commit message and git diff provided.
    Your response should state clearly in a single line what improvements or fixes were made to the code and what task has been achieved.
    The git diff provides all changes made to the code, therefore you can use it to interpret how the code changed.
    {context}
    [/INST]
"""

REDUCE_PROMPT_TEMPLATE = """
    [INST]
    The following is set of description of tasks I've done:
    {doc_summaries}
    Take these and distill it into a final, consolidated Markdown list of tasks and achievements.
    [/INST]
    Helpful Answer:
"""

CTX_TEMPLATE = """
    Git message:
    {message}

    Git diff:
    {diff}
"""


class Commit(NamedTuple):
    message: str
    diff: str


def load_model():
    # local_path = f"{os.environ['HOME']}/.cache/gpt4all/orca-mini-3b-gguf2-q4_0.gguf"
    local_path = (
        f"{os.environ['HOME']}/.cache/gpt4all/mistral-7b-instruct-v0.1.Q4_0.gguf"
    )

    # FIXME n_threads
    llm = GPT4All(
        model=local_path,
        verbose=True,
        # device="nvidia",
        n_threads=14,
        streaming=True,
    )
    return llm, ""


def get_chain():
    llm, system_prompt = load_model()

    map_prompt = PromptTemplate.from_template(
        f"{system_prompt}{dedent(MAP_PROMPT_TEMPLATE)}"
    )
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_prompt = PromptTemplate.from_template(dedent(REDUCE_PROMPT_TEMPLATE))
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_docs_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_prompt=PromptTemplate.from_template("{page_content}"),
        document_variable_name="doc_summaries",
    )

    reduce_docs_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_docs_chain,
        collapse_documents_chain=combine_docs_chain,
        token_max=1024,  # XXX
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_docs_chain,
        document_variable_name="context",
    )

    def run_llm(commits: list[Commit]) -> None:
        docs = [
            dedent(CTX_TEMPLATE.format(message=c.message, diff=c.diff)) for c in commits
        ]
        docs = [Document(page_content=doc) for doc in docs]
        print(map_reduce_chain.run(docs))
        print()

    return run_llm


def _parse_git_log(output: bytes) -> list[Commit]:
    commits: list[Commit] = []
    out_lines = output.decode("utf-8").split("\n")
    for line in out_lines:
        if not line:
            continue
        commit_hash, message = line.split(" ", maxsplit=1)
        sp = subprocess.run(
            ["git", "--no-pager", "diff", "--minimal", f"{commit_hash}^!"],
            capture_output=True,
            check=True,
        )
        diff = sp.stdout.decode("utf-8")
        commits.append(Commit(message, diff))
    return commits


def get_commits(repo_path: str) -> list[Commit]:
    with chdir(repo_path):
        # TODO add author filter
        # TODO add date filter
        sp = subprocess.run(
            [
                "git",
                "--no-pager",
                "log",
                "--no-decorate",
                "--pretty=oneline",
                "--no-merges",
                "--all",
            ],
            capture_output=True,
            check=True,
        )

        commits = _parse_git_log(sp.stdout)
        return commits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path", type=str)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    commits = get_commits(args.repo_path)
    if args.limit:
        commits = commits[: args.limit]

    chain = get_chain()
    chain(commits)


if __name__ == "__main__":
    main()
