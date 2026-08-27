"""
fetch_questions.py

GOAL OF THIS FILE
------------------
Get REAL StackOverflow questions for 4 programming-language tags
(python, r, html, css) using the official StackExchange API, and save
them to a local file (questions.json) so the rest of the project can
work offline from here on.

WHY SEPARATE THIS FROM THE REST OF THE PROJECT?
--------------------------------------------------
Fetching needs the internet and costs an API call. Clustering and
classifying (the next steps) don't need the internet at all once we
have the data -- they just read questions.json. Splitting fetching
into its own file means you only re-run this (and re-hit the API)
when you actually want fresh data, not every single time you
experiment with clustering or classification.

WHAT IS A REST API CALL, CONCRETELY?
---------------------------------------
We send an HTTP GET request (the same kind your browser sends when you
visit a webpage) to StackExchange's server, with some query parameters
telling it what we want. The server sends back data as JSON (a text
format that maps directly onto Python dicts/lists) instead of an HTML
page. `requests.get(url, params=...)` does the sending; `.json()`
parses the response text into a Python dict for us automatically.
"""

import html
import json
import requests

# The 4 tags we want labeled example questions for. This list IS the
# set of "answers" our classifier will later try to predict.
TAGS = ["python", "r", "html", "css"]

# How many questions to fetch PER TAG. 4 tags x 100 = ~400 total.
QUESTIONS_PER_TAG = 100

API_URL = "https://api.stackexchange.com/2.3/questions"


def fetch_questions_for_tag(tag: str) -> list[dict]:
    """
    Calls the StackExchange API once for a single tag and returns a
    list of plain Python dicts, one per question, containing only the
    fields we actually care about.
    """
    response = requests.get(
        API_URL,
        params={
            "site": "stackoverflow",
            "tagged": tag,
            "sort": "votes",       # highest-voted (well-established) questions first
            "order": "desc",
            "pagesize": QUESTIONS_PER_TAG,
        },
    )
    response.raise_for_status()  # crash loudly if the API call failed, instead of silently continuing
    data = response.json()

    questions = []
    for item in data["items"]:
        questions.append(
            {
                # StackExchange returns titles HTML-escaped (e.g. &quot; instead
                # of "). html.unescape() converts those back to normal text.
                "title": html.unescape(item["title"]),
                "tag": tag,          # our ground-truth label, since we searched BY this tag
                "link": item["link"],
                "score": item["score"],
            }
        )
    return questions


def main():
    all_questions = []

    for tag in TAGS:
        print(f"Fetching {QUESTIONS_PER_TAG} questions tagged '{tag}'...")
        tag_questions = fetch_questions_for_tag(tag)
        print(f"  -> got {len(tag_questions)}")
        all_questions.extend(tag_questions)

    output_path = "questions.json"
    with open(output_path, "w") as f:
        json.dump(all_questions, f, indent=2)

    print(f"\nSaved {len(all_questions)} total questions to {output_path}")


if __name__ == "__main__":
    main()
