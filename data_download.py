import csv
import random
from collections import defaultdict
from pathlib import Path

import kagglehub

from data_splitter import get_random_author_ids, get_random_posts, get_random_post_section

# settings
SAME_PRIOR = 0.5
random.seed(84208)

# string constants
SAME_AUTHOR = "Same Author"
TEXT_A = "Text A"
Text_B = "Text B"

# --------------------------------------------------------------------------------------------------

# download dataset
data_folder_path = Path("data")
try:
    data_folder_path.mkdir()
except FileExistsError:
    pass
kagglehub.dataset_download("rtatman/blog-authorship-corpus", output_dir=str(data_folder_path.resolve()))
raw_data_path = data_folder_path / "blogtext.csv"
# load data
posts: defaultdict[str, list[str]] = defaultdict(list)
with open(raw_data_path, "r", encoding="utf-8") as read_file:
    reader = csv.DictReader(read_file)

    # not a for loop; this allows for error catching
    while True:
        try:
            row = next(reader)
        # if text is too long just discard and move on
        except csv.Error as e:
            if str(e) == "field larger than field limit (131072)":
                continue
            else:
                raise e
        # break loop when end of file reached
        except StopIteration:
            break
        # no errors loading the row if we got to this point
        stripped_text: str = row["text"].strip()
        # if not between 500 and 5000 words, ignore.
        # Only 216 entries in original are > 5000 words so this mostly just saves space
        # while making sure post is long enough to be meaningful
        if 500 <= len(stripped_text.split()) < 5000:
            posts[row["id"]].append(stripped_text)

# filter out authors with only one post
single_post_authors: set[str] = set()
for author, post_list in posts.items():
    if len(post_list) == 1:
        single_post_authors.add(author)
for author in single_post_authors:
    posts.pop(author)

# get comparisons
blog_comparisons: list[dict[str, str]] = []
authors_used: set[str] = set()
# for reproducibility, it is necessary to have a guaranteed ordering of authors
authors_sorted = sorted(list(posts.keys()))
while True:
    same_author: bool = random.random() < SAME_PRIOR
    # get author(s)
    try:
        authors_current: list[str] = get_random_author_ids(authors_sorted, veto_ids=authors_used, n=(1 if same_author else 2))
    except ValueError:
        # there are not enough authors that haven't been used for the data set
        break
    authors_used.update(authors_current)
    # generate comparison
    # if I were to redo this, I would get the random post section in the original loading of the dataset.
    # however, making this change would also change the order in which pseudorandom numbers are generated and thus
    # cause the dataset to be different
    comparison = dict()
    if same_author:
        comparison[SAME_AUTHOR] = "True"
        comparison[TEXT_A], comparison[Text_B] = tuple(get_random_post_section(post)
                                                       for post in get_random_posts(posts, authors_current[0], n=2))
    else:
        comparison[SAME_AUTHOR] = "False"
        comparison[TEXT_A] = get_random_post_section(get_random_posts(posts, authors_current[0])[0])
        comparison[Text_B] = get_random_post_section(get_random_posts(posts, authors_current[1])[0])
    blog_comparisons.append(comparison)

# create test/train/dev split
random.shuffle(blog_comparisons)
dev_test_len: int = int(len(blog_comparisons) * 0.1)
train_len: int = len(blog_comparisons) - dev_test_len * 2
train: list[dict[str, str]] = blog_comparisons[:train_len]
dev: list[dict[str, str]] = blog_comparisons[train_len : train_len + dev_test_len]
test: list[dict[str, str]] = blog_comparisons[train_len + dev_test_len:]

# write to files
split_folder_path = data_folder_path / f"same-prior_{SAME_PRIOR}"
try:
    split_folder_path.mkdir()
except FileExistsError:
    pass
fieldnames: list[str] = [TEXT_A, Text_B, SAME_AUTHOR]
# train
with open(split_folder_path / "train.csv", "w", encoding="utf-8", newline="") as train_file:
    writer = csv.DictWriter(train_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(train)
# dev
with open(split_folder_path / "dev.csv", "w", encoding="utf-8", newline="") as dev_file:
    writer = csv.DictWriter(dev_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(dev)
# test
with open(split_folder_path / "test.csv", "w", encoding="utf-8", newline="") as test_file:
    writer = csv.DictWriter(test_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(test)