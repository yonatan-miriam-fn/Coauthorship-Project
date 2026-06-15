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
        text_idx = header.index("text")

        # save other rows
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
            # write reduced row to cleaned_blogtext.csv
            row[text_idx] = stripped_text
            new_row: list[str] = [row[i] for i in idx_filter]
            writer.writerow(new_row)
