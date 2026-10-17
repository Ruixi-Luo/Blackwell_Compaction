"""QASPER v0.3 可用规模核验。只做 census，尚不接入付费实验。"""
import json
import tarfile

from ..storage import ROOT, download

SOURCES = {
    "train_dev": ("https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
                  "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"),
    "test": ("https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
             "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72"),
}


def load_papers(split, offline=False):
    if split not in {"train", "dev", "test"}:
        raise ValueError("QASPER split must be train, dev or test.")
    url, sha = SOURCES["test" if split == "test" else "train_dev"]
    path = download(url, ROOT / "data/qasper" / url.rsplit("/", 1)[1], sha, offline)
    member = f"qasper-{split}-v0.3.json"
    with tarfile.open(path, "r:gz") as archive:
        papers = json.load(archive.extractfile(member))
    return papers, {"source_url": url, "source_sha256": sha, "archive_member": member,
                    "license": "CC BY 4.0", "version": "0.3"}


def build_context(paper):
    """全文候选输入：标题、摘要、所有章节和段落；不附加问答或 gold evidence。"""
    parts = [paper["title"], paper["abstract"]]
    for section in paper["full_text"]:
        parts.append(section["section_name"] or "")
        parts.extend(section["paragraphs"])
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def answer_type(answer):
    """遵循官方 evaluator 的优先级；False 是 No，不是缺失值。"""
    if answer["unanswerable"]:
        return "unanswerable"
    if answer["extractive_spans"]:
        return "extractive"
    if answer["free_form_answer"]:
        return "abstractive"
    if type(answer["yes_no"]) is bool:
        return "boolean"
    return "invalid"


def classify_question(question):
    """严格一致子集：所有标注均是 boolean 且同标签，不投票、不挑第一份答案。"""
    answers = [annotation["answer"] for annotation in question["answers"]]
    types = [answer_type(a) for a in answers]
    if not answers:
        return "no_annotations", None
    if "boolean" not in types:
        return "no_boolean", None
    if any(t != "boolean" for t in types):
        return "mixed_answer_types", None
    labels = {a["yes_no"] for a in answers}
    if len(labels) != 1:
        return "conflicting_boolean_labels", None
    return "eligible", int(next(iter(labels)))


def text_evidence_status(question, paper):
    """额外保守检查，所有标注的证据需在全文中；不裁剪正文，会引入选择偏差。"""
    evidence_lists = [a["answer"]["evidence"] for a in question["answers"]]
    if any(e.startswith("FLOAT SELECTED") for items in evidence_lists for e in items):
        return "figure_or_table"
    if not evidence_lists or any(not items for items in evidence_lists):
        return "missing_evidence"
    text = " ".join(build_context(paper).split())
    if any(not e.strip() or " ".join(e.split()) not in text for items in evidence_lists for e in items):
        return "evidence_not_in_text"
    return "text_verified"
