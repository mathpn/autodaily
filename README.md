# autodaily

`autodaily` is a command-line tool that seamlessly summarizes your daily coding activities based on your git commits. Quickly summarize your progress and plan for your daily standup meeting.

_It's completely local, therefore no code ever leaves your computer._

## Setup

1. Create a Python (>= 3.11) virtual environment

```bash
git clone https://github.com/mathpn/autodaily.git
cd autodaily
python3 -m venv ./venv
source venv/bin/activate
```

2. Ensure a recent version of git is installed

3. Install Python requirements

```bash
pip install -r requirements.txt
```

## Usage

Call `main.py` passing the paths to all repositories you want to summarize, for instance:

```bash
python main.py --repo-paths ~/Documents/repo1 ~/Documents/repo2
```

## How does it work?

An open-source [LLM](https://en.wikipedia.org/wiki/Large_language_model) is used to summarize git commits and diffs into task and achievements descriptions.

This is achieved with [Langchain](https://www.langchain.com/) through a map-reduce chain:

- Each commit is formatted to plain text (message, branch, and diff)
- Map: the LLM summarizes each commit into task descriptions and achievements
- Collapse: if all outputs of the map step combined exceed the maximum context-length, a summarization prompt is used to shorten the documents
- Reduce: the LLM combines all outputs into a single list of tasks and achievements

The pipeline above is applied iteratively to each repository.

## License

MIT
