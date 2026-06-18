import csv
import random
import re
from collections import defaultdict
from csv import DictReader
from math import ceil
from typing import Iterable

# settings
SAME_PRIOR = 0.5
random.seed(84208)

# load all the posts
posts: defaultdict[str, list[str]] = defaultdict(list)
with open("cleaned_blogtext.csv", "r", encoding="utf-8") as file:
    reader = DictReader(file)

    for row in reader:
        posts[row["id"]].append(row["text"])
# for reproducibility, it is necessary to have a guaranteed ordering at the start
authors_sorted = sorted(list(posts.keys()))

# Gets n random author ids from the authors in the dataset (default is n=1)
# Optionally can choose to veto a specific set of author ids
def get_random_author_ids(veto_ids: Iterable[str]=tuple(), n: int=1) -> list[str]:
    # remove veto_id as an option
    authors = [a for a in authors_sorted if a not in veto_ids]
    # pick a random author from the list
    return random.sample(authors, n)

# Gets n random posts from the given author (default 1)
def get_random_posts(author_id: str, n: int=1) -> list[str]:
    return random.sample(posts[author_id], n)

# Gets a random section of a blog post with the given section length (in words), default 500
def get_random_post_section(post: str, section_length: int=500) -> str:
    # split the post into words
    split_post: list[str] = re.split(r"(\s+)", post)
    # find starting point
    post_length: int = ceil(len(split_post) / 2)
    try:
        starting_point: int = random.randrange(post_length - section_length + 1) * 2
    except ValueError:
        starting_point: int = 0
    # get post section
    post_section = split_post[starting_point:starting_point + section_length * 2 - 1]
    return "".join(post_section)

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

SAME_AUTHOR = "Same Author"
TEXT_A = "Text A"
Text_B = "Text B"

# get examples
blog_comparisons: list[dict[str, str]] = []
authors_used: set[str] = set()

while True:
    same_author: bool = random.random() < SAME_PRIOR
    # get author(s)
    try:
        authors_current: list[str] = get_random_author_ids(veto_ids=authors_used, n=(1 if same_author else 2))
    except ValueError:
        # there are not enough authors that haven't been used for the data set
        break
    authors_used.update(authors_current)
    # generate comparison
    comparison = dict()
    if same_author:
        comparison[SAME_AUTHOR] = "True"
        comparison[TEXT_A], comparison[Text_B] = tuple(get_random_post_section(post)
                                                       for post in get_random_posts(authors_current[0], n=2))
    else:
        comparison[SAME_AUTHOR] = "False"
        comparison[TEXT_A] = get_random_post_section(get_random_posts(authors_current[0])[0])
        comparison[Text_B] = get_random_post_section(get_random_posts(authors_current[1])[0])
    blog_comparisons.append(comparison)

# split into test, train, dev
random.shuffle(blog_comparisons)
dev_test_len: int = int(len(blog_comparisons) * 0.1)
train_len: int = len(blog_comparisons) - dev_test_len * 2
train: list[dict[str, str]] = blog_comparisons[:train_len]
dev: list[dict[str, str]] = blog_comparisons[train_len : train_len + dev_test_len]
test: list[dict[str, str]] = blog_comparisons[train_len + dev_test_len:]

# write to files
folder_path: str = f"data_same-prior_{SAME_PRIOR}"
fieldnames: list[str] = [TEXT_A, Text_B, SAME_AUTHOR]
# train
with open(f"{folder_path}\\train.csv", "w", encoding="utf-8", newline="") as train_file:
    writer = csv.DictWriter(train_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(train)
# dev
with open(f"{folder_path}\\dev.csv", "w", encoding="utf-8", newline="") as dev_file:
    writer = csv.DictWriter(dev_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(dev)
# test
with open(f"{folder_path}\\test.csv", "w", encoding="utf-8", newline="") as test_file:
    writer = csv.DictWriter(test_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(test)