import random
import re
from math import ceil
from typing import Iterable

# Gets n random author ids from the authors in the dataset (default is n=1)
# Optionally can choose to veto a specific set of author ids
def get_random_author_ids(author_pool: Iterable[str], veto_ids: Iterable[str]=tuple(), n: int=1) -> list[str]:
    # remove veto_id as an option
    authors = [a for a in author_pool if a not in veto_ids]
    # pick a random author from the list
    return random.sample(authors, n)

# Gets n random posts from the given author (default 1)
def get_random_posts(posts: dict[str, list[str]], author_id: str, n: int=1) -> list[str]:
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