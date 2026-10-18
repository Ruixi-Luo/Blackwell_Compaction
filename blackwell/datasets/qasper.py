"""QASPER v0.3：全文输入、严格一致的二元标签、可选文本证据筛选。"""
import json
import random
import tarfile
from collections import Counter

from ..storage import ROOT, download, fingerprint

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


def convert_papers(papers, split, require_text_evidence=True):
    """证据只用于定义评测子集，不用于选择、标记或裁剪输入正文。"""
    examples, excluded = [], Counter()
    for paper_id, paper in papers.items():
        has_body = any(p.strip() for s in paper['full_text'] for p in s['paragraphs'])
        context = build_context(paper)
        for question in paper['qas']:
            status, label = classify_question(question)
            if status == 'eligible' and not has_body:
                status = 'missing_full_text'
            if status != 'eligible':
                excluded[status] += 1
                continue
            evidence = text_evidence_status(question, paper)
            if require_text_evidence and evidence != 'text_verified':
                excluded[evidence] += 1
                continue
            examples.append({
                'id': f"qasper-{split}-{question['question_id']}",
                'paper_id': paper_id, 'context': context,
                'context_group': fingerprint(context),
                'question': question['question'], 'label': label,
                'evidence_status': evidence,
            })
    if len({e['id'] for e in examples}) != len(examples):
        raise ValueError('Duplicate QASPER question IDs.')
    return examples, dict(excluded)


def load_qasper(split, sample_size, seed, offline=False):
    papers, source = load_papers(split, offline)
    pool, excluded = convert_papers(papers, split, require_text_evidence=True)
    if not 1 <= sample_size <= len(pool):
        raise ValueError('sample_size exceeds the eligible QASPER text-evidence pool.')
    indices = random.Random(seed).sample(range(len(pool)), sample_size)
    provenance = {
        **source, 'name': 'QASPER yes/no (text evidence)', 'split': split,
        'dataset_home': 'https://allenai.org/data/qasper',
        'available_rows': len(pool), 'excluded': excluded, 'sample_seed': seed,
        'selection_rule': 'Nonempty body; all annotations boolean and consistent; every annotation has nonempty evidence found in the text; no figure/table evidence.',
        'context_definition': 'Title, abstract and all sections/paragraphs in order. No questions, answers or evidence markers appended.',
        'scope': 'Exploratory use of train; dev/test untouched. Evidence filtering changes label and task distribution. Each paper is compressed once and reused across its questions.',
    }
    return [pool[i] for i in indices], provenance
