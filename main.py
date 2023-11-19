import argparse
import os
import subprocess
from contextlib import chdir
from textwrap import dedent
from typing import NamedTuple

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate


class Commit(NamedTuple):
    message: str
    diff: str


def get_chain():
    # local_path = f"{os.environ['HOME']}/.cache/gpt4all/orca-mini-3b-gguf2-q4_0.gguf"
    local_path = (
        f"{os.environ['HOME']}/.cache/gpt4all/mistral-7b-instruct-v0.1.Q4_0.gguf"
    )
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    SYSTEM_PROMPT = ""

    PROMPT_TEMPLATE = """
    [INST]
    Generate a concise task description in past tense based on the git commit message and git diff provided.
    Your response should state clearly in a single line what improvements were made to the code and what task has been achieved.
    The git diff provides all changes made to the code, therefore you can use it to interpret how the code changed.

    Git message:
    {message}

    Git diff:
    {diff}

    [/INST]
    """

    # SYSTEM_PROMPT = "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n"
    #
    # PROMPT_TEMPLATE = """
    # ### System:
    # Generate a concise task description in past tense based on the git commit message and git diff provided.
    # Your response should state clearly in a single line what improvements were made to the code and what task has been achieved.
    # The git diff provides all changes made to the code, therefore you can use it to interpret how the code changed.
    #
    # ### User
    # Git message:
    # {message}
    #
    # Git diff:
    # {diff}
    #
    # ### Response:\n
    # """

    prompt = PromptTemplate(
        template=f"{SYSTEM_PROMPT}{PROMPT_TEMPLATE}", input_variables=["commits"]
    )

    # FIXME n_threads
    llm = GPT4All(
        model=local_path,
        callback_manager=callback_manager,
        verbose=True,
        # device="nvidia",
        n_threads=14,
        streaming=True,
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    def run_llm(commit: Commit) -> None:
        diff = dedent(commit.diff)
        message = dedent(commit.message)
        # print(prompt.format(diff=diff, message=message))
        llm_chain.run(diff=diff, message=message)
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
    for commit in commits:
        print("--------------------------------")
        print(f"-> {commit.message}")
        chain(commit)


if __name__ == "__main__":
    main()
