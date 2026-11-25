# expect: none
def first_sentence(text: str) -> str:
    t = " ".join(text.split())
    for i, ch in enumerate(t):
        if ch == "." and (i + 1 == len(t) or t[i + 1] == " "):
            return t[: i + 1]
    return t


def split_camel(name: str) -> list:
    out, cur = [], ""
    for i in range(len(name)):
        cur += name[i]
        if i + 1 < len(name) and name[i + 1].isupper():
            out.append(cur)
            cur = ""
    return out + [cur]


print(first_sentence("Prices are adjusted. Data starts in 2000."), split_camel("closePriceAdj"))
