import json, os

class ArtifactWriter:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def write(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n"); self.f.flush()

    def close(self):
        self.f.close()
