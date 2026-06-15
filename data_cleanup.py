import csv

# filter data from blogtext.csv
with open("blogtext.csv", "r", encoding="utf-8") as read_file:
    with open("cleaned_blogtext.csv", "w", encoding="utf-8", newline="") as write_file:
        reader = csv.reader(read_file)
        writer = csv.writer(write_file)

        # save header
        header: list[str] = next(reader)
        cleaned_header: list[str] = ["id", "text"]
        writer.writerow(cleaned_header)
        idx_filter = [header.index(item) for item in cleaned_header]
        id_idx = header.index("id")
        text_idx = header.index("text")

        # save other rows
        unique_entry: dict[str, list[str]] = dict()
        seen_authors: set[str] = set()
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
            stripped_text: str = row[text_idx].strip()
            # if not between 500 and 5000 words, ignore.
            # Only 216 entries in original are > 5000 words so this mostly just saves space
            # while making sure post is long enough to be meaningful
            if not 500 <= len(stripped_text.split()) < 5000:
                continue
            # make row to put in cleaned file
            row[text_idx] = stripped_text
            new_row: list[str] = [row[i] for i in idx_filter]
            # only add entry if author has been seen before (to avoid single-post-authors)
            author_id: str = row[id_idx]
            if author_id in seen_authors:
                # if this is the second entry for an author, write the first entry to file first
                if author_id in unique_entry:
                    writer.writerow(unique_entry.pop(author_id))
                # write this entry to the file
                writer.writerow(new_row)
            else:
                # ensure that the author is saved and that the first entry is as well in case of a second
                seen_authors.add(author_id)
                unique_entry[author_id] = new_row
