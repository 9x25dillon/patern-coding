"""Fixed UTF-8 byte vocabulary. No downloads, fitting, or unknown tokens."""


class ByteTokenizer:
    pad_id, bos_id, eos_id, sep_id = 0, 1, 2, 3
    vocab_size = 260
    version = "utf8-byte-v1"

    def encode(self, text: str) -> list[int]:
        return [b + 4 for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        return bytes(i - 4 for i in ids if 4 <= i < 260).decode("utf-8", errors="replace")

    def example(self, record: dict) -> tuple[list[int], list[bool]]:
        if "prompt" in record and "completion" in record:
            prefix = [self.bos_id] + self.encode(record["prompt"]) + [self.sep_id]
            answer = self.encode(record["completion"]) + [self.eos_id]
            return prefix + answer, [False] * len(prefix) + [True] * len(answer)
        body = self.encode(record["text"]) + [self.eos_id]
        return [self.bos_id] + body, [False] + [True] * len(body)
