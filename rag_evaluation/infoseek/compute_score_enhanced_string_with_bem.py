"""
InfoSeek Evaluation Script.
Credits: https://github.com/edchengg/infoseek_eval/blob/main/infoseek_eval.py
"""

import io
import os
import re
import sys
import json
import importlib
import ujson
import string
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
import argparse
from tqdm import tqdm
from os.path import isfile, join
from os import listdir
def normalize_answer(text: str) -> str:
    """Normalize a given text by removing articles, punctuation, and white spaces, and converting to lowercase."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lowercase(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lowercase(text))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if the normalized prediction exactly matches the normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def canonical_numeric(value: Union[str, float, int]) -> str:
    """Convert numeric answers to a canonical string representation."""
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(round(float(value))))
        return format(float(value), '.15g')
    return str(value)


def relaxed_match_score(prediction: str, ground_truth: str) -> bool:
    """Relaxed matching for string questions; falls back to exact match."""
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)

    if norm_pred == norm_gt:
        return True

    if not norm_pred or not norm_gt:
        return False

    if not any(ch.isalpha() for ch in norm_gt):
        return False

    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    if not gt_tokens:
        return False

    if all(token in pred_tokens for token in gt_tokens):
        return True

    if len(gt_tokens) > 1:
        window = len(gt_tokens)
        for idx in range(len(pred_tokens) - window + 1):
            if pred_tokens[idx:idx + window] == gt_tokens:
                return True

    return False


def metric_max_over_ground_truths(
    metric_fn,
    prediction: str,
    ground_truths: List[str]
) -> Union[int, bool]:
    """Compute the maximum score of a prediction over a list of ground truths using a given metric function."""
    return max(
        metric_fn(prediction, ground_truth) for ground_truth in ground_truths
    )


def in_range(number: float, range_list: Tuple[float, float]) -> bool:
    """Check if a number is within the specified range (inclusive)."""
    min_num, max_num = range_list
    return min_num <= number <= max_num


def safe_division(x: float, y: float) -> float:
    """Divide x by y, returning 0 if y is 0."""
    return x / y if y != 0 else 0


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_BEM_EVAL_ROOT = (
    os.environ.get("INFOSEEK_BEM_EVAL_ROOT")
    or os.environ.get("BEM_EVAL_ROOT")
    or os.path.join(_REPO_ROOT, "rag_evaluation", "evqa_eval")
)
_BEM_EVALUATION_FUNCTION: Optional[Callable[[Dict[str, Any]], float]] = None
_BEM_INIT_ERROR: Optional[Exception] = None
_BEM_RUNTIME_WARNING_EMITTED = False
QUESTION_TEXT_KEYS = (
    'question',
    'question_text',
    'question_str',
    'question_template',
    'question_with_entity',
    'query',
    'prompt',
)


class LogFriendlyTqdmFile(io.TextIOBase):
    """Wrap a stream so tqdm writes break onto new lines for log tailing."""

    def __init__(self, stream: io.TextIOBase):
        super().__init__()
        self._stream = stream

    def write(self, data: str) -> int:
        if not data:
            return 0
        formatted = data.replace('\r', '\r\n')
        self._stream.write(formatted)
        self._stream.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()


def resolve_progress_stream() -> io.TextIOBase:
    """Return a stream suitable for tqdm, friendlier for non-interactive logs."""
    stream = getattr(sys, 'stderr', None) or getattr(sys, 'stdout', None)
    if stream is None:
        raise RuntimeError('No stderr/stdout available for progress reporting.')
    try:
        is_tty = stream.isatty()
    except Exception:
        is_tty = False
    if is_tty:
        return stream
    return LogFriendlyTqdmFile(stream)


def log_progress(message: str) -> None:
    """Write a progress message to the same stream used for tqdm."""
    stream = resolve_progress_stream()
    stream.write(message + '\n')
    stream.flush()


def iter_with_progress(
    iterable,
    *,
    enable: bool = False,
    desc: Optional[str] = None,
):
    """Return iterable wrapped with tqdm when progress display is enabled."""
    if not enable:
        return iterable
    total = None
    try:
        total = len(iterable)  # type: ignore[arg-type]
    except TypeError:
        total = None
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=False,
        file=resolve_progress_stream(),
        dynamic_ncols=True,
    )


def map_to_bem_question_type(question_type: Optional[str]) -> str:
    """Map InfoSeek question types to the limited BEM schema."""
    if not question_type:
        return 'templated'
    normalized = str(question_type).lower()
    if 'multi' in normalized:
        return 'multi_answer'
    return 'templated'


def get_bem_evaluation_function() -> Optional[Callable[[Dict[str, Any]], float]]:
    """Lazily load the BEM evaluation function used for string scoring."""
    global _BEM_EVALUATION_FUNCTION, _BEM_INIT_ERROR
    if _BEM_EVALUATION_FUNCTION is not None:
        return _BEM_EVALUATION_FUNCTION
    if _BEM_INIT_ERROR is not None:
        return None

    eval_root = DEFAULT_BEM_EVAL_ROOT
    if eval_root:
        eval_root = os.path.abspath(os.path.expanduser(eval_root))
        if eval_root not in sys.path:
            sys.path.append(eval_root)

    try:
        evqa_utils = importlib.import_module('evqa_utils')
        initialize_encyclopedic_vqa_evaluation_function = (
            evqa_utils.initialize_encyclopedic_vqa_evaluation_function
        )
    except ImportError as exc:  # pragma: no cover - dependency check
        _BEM_INIT_ERROR = exc
        detail = f'{exc.__class__.__name__}: {exc}'
        if isinstance(exc, ModuleNotFoundError) and exc.name == 'evqa_utils':
            detail += f' (looked in BEM eval root: {eval_root})'
        elif isinstance(exc, ModuleNotFoundError) and exc.name:
            detail += (
                f' (missing dependency `{exc.name}` required by evqa_utils; '
                'install TensorFlow/TensorFlow Hub/Text dependencies to enable BEM)'
            )
        print(
            'Warning: failed to import evqa_utils for BEM scoring. '
            f'{detail}. Falling back to exact string matching.'
        )
        return None

    try:
        _BEM_EVALUATION_FUNCTION = initialize_encyclopedic_vqa_evaluation_function()
    except Exception as exc:  # pragma: no cover - dependency check
        _BEM_INIT_ERROR = exc
        print(
            f'Warning: failed to initialize BEM evaluation function ({exc}). '
            'Falling back to exact string matching.'
        )
        return None

    return _BEM_EVALUATION_FUNCTION


def extract_question_text(example: Dict[str, Any]) -> str:
    """Retrieve the most useful question text available for BEM scoring."""
    for key in QUESTION_TEXT_KEYS:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ''


def coerce_answer_list(answer: Any) -> List[str]:
    """Convert answer annotations to a flat list of strings."""
    if isinstance(answer, list):
        values = answer
    elif answer is None:
        values = []
    else:
        values = [answer]

    normalized: List[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            for field in ('value', 'wikidata', 'text', 'answer'):
                if value.get(field) is not None:
                    normalized.append(str(value[field]))
                    break
            else:
                normalized.append(str(value))
        else:
            normalized.append(str(value))
    return normalized


def evaluate_string_prediction(
    question_text: str,
    references: List[str],
    prediction: Optional[str],
    *,
    bem_question_type: str = 'templated',
    data_id: Optional[str] = None,
) -> bool:
    """Evaluate a string prediction using BEM (with exact-match fallback)."""
    candidate = '' if prediction is None else str(prediction).strip()
    if not candidate:
        return False
    references_clean = [ref.strip() for ref in references if isinstance(ref, str) and ref.strip()]
    if not references_clean:
        return False

    bem_fn = get_bem_evaluation_function()
    if bem_fn is None:
        return bool(
            metric_max_over_ground_truths(
                exact_match_score,
                candidate,
                references_clean,
            )
        )

    example_base = {
        'question': question_text or '',
        'candidate': candidate,
        'question_type': bem_question_type,
    }
    for reference in references_clean:
        example = dict(example_base)
        example['reference'] = reference
        try:
            bem_score = bem_fn(example)
        except Exception as exc:  # pragma: no cover - tensorflow errors
            global _BEM_RUNTIME_WARNING_EMITTED
            if not _BEM_RUNTIME_WARNING_EMITTED:
                warn_id = f' (data_id={data_id})' if data_id else ''
                print(
                    f'Warning: BEM scoring failed{warn_id}: {exc}. '
                    'Falling back to exact string matching.'
                )
                _BEM_RUNTIME_WARNING_EMITTED = True
            return bool(
                metric_max_over_ground_truths(
                    exact_match_score,
                    candidate,
                    references_clean,
                )
            )
        if bem_score >= 0.5:
            return True
    return False


def metric_numerical_range(
    pred: Union[float, Tuple[float, float], List[float]],
    answer: Union[float, Tuple[float, float], List[float]],
    tolerance: float = 0.0,
) -> int:
    """Scores numerical questions based on ranges and tolerances.

    1) First, convert single number answer to a range with +/- tolerance.
    2) If prediction is a single number, return 1 if it's in the answer range, 0
    otherwise.
    3) If prediction is a range, return 1 if the range is in the answer range or
    if the IOU
        (overlap between prediction and answer range) > 0.5, 0 otherwise.

    Args:
        pred: A list/tuple of 2 numbers or a single number.
        answer: A list/tuple of 2 numbers or a single number.
        tolerance: A float value for the tolerance range (default: 0.1).

    Returns:
        int: 1 if conditions are met, 0 otherwise.
    """
    answer = list(answer) if isinstance(answer, tuple) else answer
    pred = list(pred) if isinstance(pred, tuple) else pred

    if not isinstance(answer, list):
        answer = [answer * (1 - tolerance), answer * (1 + tolerance)]

    # Prediction is a single number
    if not isinstance(pred, list):
        return 1 if in_range(pred, answer) else 0

    # Prediction is a range
    if answer[0] <= pred[0] <= answer[1] and answer[0] <= pred[1] <= answer[1]:
        return 1
    else:
        iou = range_intersection_over_union(pred, answer)
        return 1 if iou >= 0.5 - 1e-12 else 0


def process_numerical_answer(string_number: str) -> Union[float, List[float]]:
    """Parses numerical answer string into numbers (a single number or a range).

    1) Clean the string and extract numbers;
    2) if there are 2 numbers, return a range as [minimum value, maximum value]
        else if there is 1 number, return a single number
        else return [0, 0]

    Args:
        string_number: A string representing a numerical answer.

    Returns:
        A single digit or a list with 2 numbers.
    """
    # Clean string
    string_number = clean_str_range(string_number)
    numerical_numbers_tmp = re.findall(
        r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', string_number
    )
    numerical_numbers_tmp = [
        n.replace(',', '').strip('.') for n in numerical_numbers_tmp
    ]
    numerical_numbers = []
    for n in numerical_numbers_tmp:
        if n.count('.') > 1:
            n = n.split('.')[0]
            numerical_numbers.append(float(n))
        else:
            numerical_numbers.append(float(n))

    # Use the first 2 numbers
    if len(numerical_numbers) > 2:
        numerical_numbers = numerical_numbers[:2]

    if len(numerical_numbers) == 2:
        first_val = numerical_numbers[0]
        second_val = numerical_numbers[1]
        return [first_val, second_val] if first_val <= second_val else first_val
    elif len(numerical_numbers) == 1:
        return numerical_numbers[0]
    else:
        return [0, 0]


def find_all(s: str, c: str) -> Generator[int, None, None]:
    """Find all occurrences of a character in a string and return their indices.

    Args:
        s: The input string to search.
        c: The character to search for.

    Yields:
        int: The index of the next occurrence of the character.
    """
    idx = s.find(c)
    while idx != -1:
        yield idx
        idx = s.find(c, idx + 1)


def clean_str_range(text: str) -> str:
    """Clean range expression in a string (e.g., '9-10' --> '9 - 10').

    Args:
        text: The input string containing the range expression.

    Returns:
        str: The cleaned string with proper spacing around the hyphen.
    """
    idx_list = list(find_all(text, '-'))
    idx_replace = [
        idx for idx in idx_list if idx >= 1 and text[idx - 1].isdigit()
    ]
    new_str = ''.join(
        ' - ' if idx in idx_replace else s for idx, s in enumerate(text)
    )
    return new_str


def range_intersection_over_union(
    x_list: List[float], y_list: List[float]
) -> float:
    """Calculate the intersection over union (IOU) of two ranges."""
    min_1, max_1 = min(x_list), max(x_list)
    min_2, max_2 = min(y_list), max(y_list)

    overlap = max(0.0, min(max_1, max_2) - max(min_1, min_2))
    length_x = (max_1 - min_1) + 1e-12
    length_y = (max_2 - min_2) + 1e-12
    iou = safe_division(overlap, length_x + length_y - overlap)
    return iou


def evaluate_quantity(
    quantity_pred: List[str],
    quantity_answer: List[List[str]],
) -> List[int]:
    """Evaluate numerical predictions against numerical answers."""
    return [
        metric_max_over_ground_truths(
            exact_match_score,
            canonical_numeric(pred),
            [canonical_numeric(a) for a in ans],
        )
        for pred, ans in zip(quantity_pred, quantity_answer)
    ]


def evaluate_entity(entity_entries: List[Dict[str, Any]]) -> List[int]:
    """Evaluate string/entity predictions using BEM (with exact fallback)."""
    results: List[int] = []
    index = 0
    for entry in entity_entries:
        question_text = entry.get('question') or ''
        prediction = entry.get('prediction')
        answers = entry.get('answers', [])
        bem_qtype = entry.get('bem_question_type', 'templated')
        data_id = entry.get('data_id')
        results.append(
            int(
                evaluate_string_prediction(
                    question_text,
                    answers,
                    prediction,
                    bem_question_type=bem_qtype,
                    data_id=data_id,
                )
            )
        )
        print(f'Entity evaluation {index}: prediction="{prediction}", answers={answers}')
        index += 1
    return results


def evaluate_time(
    time_pred: List[str], time_answer: List[List[str]]
) -> List[int]:
    """Evaluate time predictions against time answers.

    Criteria:
    1) +/- one year --> correct
    2) if asking for date, but the year is correct --> correct

    Args:
        time_pred: prediction of time
        time_answer: a list of time reference

    Returns:
        List: 0 or 1
    """
    return [
        metric_max_over_ground_truths(exact_match_score, pred, ans)
        for pred, ans in zip(time_pred, time_answer)
    ]


def evaluation(
    predictions: List[Dict[str, Any]],
    qid2example: Dict[str, Dict[str, Any]],
    *,
    show_progress: bool = False,
    progress_label: Optional[str] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Evaluate predictions against ground truth answers.

    Separate questions into time, numerical, and string categories.

    Args:
        predictions: A list of predictions.
        qid2example: A mapping from question ID to ground truth examples.

    Returns:
        Tuple[List[int], List[int], List[int]]: Lists of scores for time,
        quantity, and entity predictions.
    """
    time_pred, quantity_pred = [], []
    time_answer, quantity_answer = [], []
    entity_entries: List[Dict[str, Any]] = []

    iterator = iter_with_progress(
        predictions,
        enable=show_progress,
        desc=progress_label,
    )

    for p in iterator:
        quid = p['data_id']
        if quid not in qid2example:
            continue
        example = qid2example[quid]
        pred = p['prediction']
        print(f'Prediction: {pred}')
        answer = example['answer_eval']
        question_type = example['question_type'].lower()
        print(f'Question type: {question_type}')
        if question_type == 'time':
            time_pred.append(pred)
            time_answer.append(answer)
        elif question_type == 'numerical':
            quantity_pred.append(pred)
            quantity_answer.append([canonical_numeric(a) for a in answer])
        else:
            entity_entries.append(
                {
                    'prediction': pred,
                    'answers': coerce_answer_list(answer),
                    'question': extract_question_text(example),
                    'bem_question_type': map_to_bem_question_type(question_type),
                    'data_id': quid,
                }
            )

    score_time = evaluate_time(time_pred, time_answer)
    score_quantity = evaluate_quantity(quantity_pred, quantity_answer)
    print(f'Start evaluating entity Entity entries to evaluate: {len(entity_entries)}')
    score_entity = evaluate_entity(entity_entries)
    return score_time, score_quantity, score_entity


def get_results(
    predictions: List[Dict[str, Any]],
    qid2example: Dict[str, Dict[str, Any]],
    *,
    show_progress: bool = False,
    progress_label: Optional[str] = None,
) -> Tuple[float, float, float, float]:
    """Get evaluation scores for predictions.

    Args:
        predictions: A list of predictions.
        qid2example: A mapping from question ID to ground truth examples.

    Returns:
        Tuple[float, float, float, float]: Final scores for time, quantity,
        entity, and overall predictions.
    """
    score_time, score_quantity, score_entity = evaluation(
        predictions,
        qid2example,
        show_progress=show_progress,
        progress_label=progress_label,
    )
    final_score_time = safe_division(sum(score_time), len(score_time))
    final_score_quantity = safe_division(
        sum(score_quantity), len(score_quantity))
    final_score_entity = safe_division(sum(score_entity), len(score_entity))
    final_score = safe_division(
        sum(score_time + score_quantity + score_entity),
        len(score_time + score_quantity + score_entity),
    )
    return final_score, final_score_time, final_score_quantity, final_score_entity


def harmonic_mean(*args: float) -> float:
    """Calculate the harmonic mean of the input arguments."""
    args_safe = [a if a != 0 else 1e-12 for a in args]
    hmean = len(args_safe) / sum((1.0 / val) for val in args_safe)
    return hmean


def evaluate_infoseek(
    predictions: List[Dict[str, Any]],
    qid2example: Dict[str, Dict[str, Any]],
    *,
    show_progress: bool = False,
    progress_label: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate predictions against references.

    Args:
        predictions: A list of predictions.
        qid2example: A dictionary of reference with question_id as key.

    Returns:
        Dict[str, float]: A dictionary containing the final scores for time,
        quantity, entity, and overall predictions.
    """
    final_score, score_time, score_num, score_string = get_results(
        predictions,
        qid2example,
        show_progress=show_progress,
        progress_label=progress_label,
    )
    return {
        'score': round(final_score * 100, 2),
        'score_time': round(score_time * 100, 2),
        'score_num': round(score_num * 100, 2),
        'score_string': round(score_string * 100, 2),
    }


def evaluate_infoseek_full(
    predictions: List[List[Dict[str, Any]]],
    qid2examples: List[Dict[str, Dict[str, Any]]],
    *,
    show_progress: bool = False,
    split_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    infoseek_score = []
    total_predictions = sum(len(pred_list) for pred_list in predictions)
    print(f'Total predictions to evaluate: {total_predictions}')
    processed_predictions = 0
    for idx, (pred, qid2example) in enumerate(zip(predictions, qid2examples)):
        label = (
            split_labels[idx]
            if split_labels and idx < len(split_labels)
            else f'split_{idx + 1}'
        )
        if show_progress and total_predictions:
            start_idx = processed_predictions + 1 if pred else processed_predictions
            end_idx = processed_predictions + len(pred)
            log_progress(
                f"[evaluate_infoseek_full] processing predictions {start_idx}-"
                f"{end_idx} / {total_predictions} ({label})"
            )
        processed_predictions += len(pred)
        split_score = evaluate_infoseek(
            pred,
            qid2example,
            show_progress=show_progress,
            progress_label=f'Evaluating {label}',
        )
        infoseek_score.append(split_score)
        print(idx)
    split_scores = [score['score'] for score in infoseek_score]
    return {
        'final_score': round(harmonic_mean(*split_scores), 2),
        'unseen_question_score': infoseek_score[0],
        'unseen_entity_score': infoseek_score[1],
    }


def evaluate(prediction_path: str, reference_path: str, reference_qtype_path: str, adjust_score: bool) -> Dict[str, Any]:
    """Evaluate predictions against references.

    Args:
        prediction_path: Path to prediction file.
        reference_path: Path to reference file.
        reference_qtype_path: Path to reference question type file.

    Returns:
        Dict[str, Any]: A dictionary containing the final scores for time,
        quantity, entity, and overall predictions.
    """
    predictions = load_jsonl(prediction_path)
    reference = load_jsonl(reference_path)
    reference_qtype = load_jsonl(reference_qtype_path)
    if adjust_score:
        ids = set(x['data_id'] for x in predictions)
        reference = [x for x in reference if x['data_id'] in ids]
        reference_qtype = [x for x in reference_qtype if x['data_id'] in ids]
    qid2example = prepare_qid2example(reference, reference_qtype)
    # split predictions into two splits: unseen_question and unseen_entity
    unseen_question = []
    unseen_entity = []
    for pred in predictions:
        print(pred)
        data_id = pred['data_id']
        if data_id in qid2example:
            if qid2example[data_id]['data_split'].endswith('unseen_question'):
                unseen_question.append(pred)
            else:
                unseen_entity.append(pred)
        else:
            pass
    return evaluate_infoseek_full([unseen_question, unseen_entity], [qid2example, qid2example])


def prepare_qid2example(
    reference: List[Dict[str, Any]],
    reference_qtype: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Convert reference to qid2example dictionary."""
    qid2qtype = dict()
    for qtype in reference_qtype:
        qid = qtype["data_id"]
        qid2qtype[qid] = qtype["question_type"]

    qid2example = dict()
    for r in reference:
        qid = r['data_id']
        q_type = qid2qtype[qid].lower()
        if q_type == 'numerical':
            answer_eval = r.get('answer_eval')
            if isinstance(answer_eval, list):
                ans_entry = answer_eval[0] if answer_eval else {}
            else:
                ans_entry = answer_eval

            wikidata_val = None
            if isinstance(ans_entry, dict):
                wikidata_val = ans_entry.get('wikidata')

            if wikidata_val is not None:
                r['answer_eval'] = [canonical_numeric(wikidata_val)]
            elif isinstance(ans_entry, dict):
                if 'value' in ans_entry and ans_entry['value'] is not None:
                    r['answer_eval'] = [canonical_numeric(ans_entry['value'])]
                elif 'range' in ans_entry and ans_entry['range']:
                    r['answer_eval'] = [
                        canonical_numeric(ans_entry['range'][0])
                    ]
                else:
                    r['answer_eval'] = [canonical_numeric(ans_entry)]
            else:
                r['answer_eval'] = [canonical_numeric(ans_entry)]

        qid2example[qid] = r
        qid2example[qid]["question_type"] = qid2qtype[qid]
    return qid2example


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of Dict[strionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(ujson.loads(line))
    return data


def load_entity_metadata(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load entity metadata (id/text) keyed by data_id."""
    if not path:
        return {}
    if not os.path.exists(path):
        print(f'Warning: entity metadata file not found at {path}')
        return {}
    metadata: Dict[str, Dict[str, Any]] = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                entry = ujson.loads(line)
            except ValueError:
                continue
            data_id = entry.get('data_id')
            if not data_id:
                continue
            metadata[data_id] = {
                'entity_id': entry.get('entity_id'),
                'entity_text': entry.get('entity_text'),
            }
    return metadata


def normalize_for_entity(text: Optional[str]) -> str:
    """Normalize entity-related text for comparison."""
    if text is None:
        return ''
    normalized = normalize_answer(text)
    return normalized.strip()


def title_matches_entity(
    selected_title: Optional[str],
    entity_text: Optional[str],
) -> Optional[bool]:
    """Determine whether the selected title matches the ground-truth entity."""
    if not selected_title or not entity_text:
        return None

    title_norm = normalize_for_entity(selected_title)
    entity_norm = normalize_for_entity(entity_text)
    if not title_norm or not entity_norm:
        return None

    return (
        entity_norm in title_norm
        or title_norm in entity_norm
    )


def ensure_merge_file(pred_path: str) -> str:
    """Ensure that a merge.jsonl file exists for the given prediction directory."""
    merge_data = os.path.join(pred_path, 'merge.jsonl')
    if os.path.exists(merge_data):
        return merge_data
    
    if pred_path.endswith('.jsonl'):
        return pred_path

    files = [
        os.path.join(pred_path, f)
        for f in listdir(pred_path)
        if isfile(join(pred_path, f)) and ('results' not in f)
    ]
    data: List[Dict[str, Any]] = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                single_file = ujson.load(f)
            if isinstance(single_file, list):
                data.extend(single_file)
        except Exception as exc:
            print(f'Failed to open {file_path}: {exc}')

    with open(merge_data, 'w', encoding='utf-8') as f:
        for entry in data:
            ujson.dump(entry, f)
            f.write('\n')
    return merge_data


def filter_qid2example_by_ids(
    qid2example: Dict[str, Dict[str, Any]],
    ids: Optional[Set[str]],
) -> Dict[str, Dict[str, Any]]:
    """Filter qid2example dictionary by a set of allowed IDs."""
    if not ids:
        return qid2example
    return {qid: qid2example[qid] for qid in ids if qid in qid2example}


def evaluate_predictions_with_qid(
    predictions: List[Dict[str, Any]],
    qid2example: Dict[str, Dict[str, Any]],
    *,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Evaluate predictions given a prepared qid2example mapping."""
    unseen_question = []
    unseen_entity = []
    for pred in predictions:
        data_id = pred.get('data_id')
        if not data_id or data_id not in qid2example:
            continue
        split = qid2example[data_id].get('data_split', '')
        if split.endswith('unseen_question'):
            unseen_question.append(pred)
        else:
            unseen_entity.append(pred)
    print(f'Unseen question count: {len(unseen_question)}')
    print(f'Unseen entity count: {len(unseen_entity)}')
    return evaluate_infoseek_full(
        [unseen_question, unseen_entity],
        [qid2example, qid2example],
        show_progress=show_progress,
        split_labels=['unseen_question', 'unseen_entity'],
    )


def index_predictions_by_id(
    predictions: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Index predictions by their data_id."""
    indexed: Dict[str, Dict[str, Any]] = {}
    for pred in predictions:
        data_id = pred.get('data_id')
        if data_id:
            indexed[data_id] = pred
    return indexed


def score_method_prediction(
    example: Dict[str, Any],
    prediction_entry: Optional[Dict[str, Any]],
) -> Tuple[int, Optional[Union[float, List[float], str]]]:
    """Score a single prediction entry against the provided example."""
    if not prediction_entry:
        return 0, None

    prediction_text = prediction_entry.get('prediction')
    if prediction_text is None or str(prediction_text).strip() == '':
        return 0, None

    question_type = str(example.get('question_type', '')).lower()
    answer_eval = example.get('answer_eval', [])

    if question_type == 'numerical':
        processed_prediction = canonical_numeric(prediction_text)
        canonical_answers = [canonical_numeric(ans) for ans in answer_eval]
        score = metric_max_over_ground_truths(
            exact_match_score,
            processed_prediction,
            canonical_answers,
        )
        return int(score), processed_prediction

    processed_prediction = normalize_answer(str(prediction_text))
    if question_type == 'time':
        normalized_answers = [normalize_answer(str(ans)) for ans in answer_eval]
        score = metric_max_over_ground_truths(
            exact_match_score,
            processed_prediction,
            normalized_answers,
        )
        return int(score), processed_prediction

    string_answers = coerce_answer_list(answer_eval)
    question_text = extract_question_text(example)
    is_correct = evaluate_string_prediction(
        question_text,
        string_answers,
        prediction_text,
        bem_question_type=map_to_bem_question_type(question_type),
        data_id=example.get('data_id'),
    )
    return int(is_correct), processed_prediction


def build_method_view(
    prediction_entry: Optional[Dict[str, Any]],
    is_correct: int,
    processed_prediction: Optional[Union[float, List[float], str]],
    *,
    include_selected_title: bool = True,
    include_selected_url: bool = True,
) -> Dict[str, Any]:
    """Build a detailed view of a method's prediction for analysis output."""
    view: Dict[str, Any] = {
        'available': prediction_entry is not None,
        'correct': bool(is_correct),
        'prediction': None,
        'normalized_prediction': None,
        'processed_prediction': processed_prediction,
    }

    if prediction_entry:
        raw_prediction = prediction_entry.get('prediction')
        view['prediction'] = raw_prediction
        view['normalized_prediction'] = (
            normalize_answer(str(raw_prediction))
            if raw_prediction is not None else None
        )
        if include_selected_url:
            selected_url = prediction_entry.get('selected_url')
            if selected_url is not None:
                view['selected_url'] = selected_url
        if include_selected_title:
            selected_title = prediction_entry.get('selected_title')
            if selected_title is not None:
                view['selected_title'] = selected_title
        section_text = prediction_entry.get('selected_section_text')
        if section_text is not None:
            view['selected_section_text'] = section_text
    return view


def slugify(value: str) -> str:
    """Convert a string to a filesystem-friendly slug."""
    value = value.strip().replace(' ', '_')
    value = re.sub(r'[^0-9a-zA-Z_]+', '_', value)
    value = re.sub(r'_+', '_', value)
    return value.strip('_') or 'method'


def write_jsonl(path: str, entries: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(path, 'w', encoding='utf-8') as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii=False))
            file.write('\n')


def run_case_analysis(
    method_a: Dict[str, Any],
    method_b: Dict[str, Any],
    qid2example: Dict[str, Dict[str, Any]],
    output_root: str,
) -> None:
    """Run a detailed case analysis comparing two methods."""
    if not qid2example:
        print('No reference examples available for case analysis.')
        return

    method_a_name = method_a['name']
    method_b_name = method_b['name']

    pair_slug = slugify(f"{method_a_name}_vs_{method_b_name}")
    analysis_dir = os.path.join(output_root, pair_slug)
    os.makedirs(analysis_dir, exist_ok=True)

    def dynamic_bucket_key(*parts: str) -> str:
        """Create a filesystem-friendly bucket key from multiple parts."""
        combined = '|'.join(parts)
        return slugify(combined) or 'bucket'

    method_a_only_key = f'only_{slugify(method_a_name)}'
    method_b_only_key = f'only_{slugify(method_b_name)}'

    general_buckets: Dict[str, List[Dict[str, Any]]] = {
        'both_correct': [],
        method_a_only_key: [],
        method_b_only_key: [],
        'both_incorrect': [],
    }
    detailed_buckets: Dict[str, List[Dict[str, Any]]] = {}
    method_stats: Dict[str, Dict[str, int]] = {
        method_a_name: {'total': 0, 'correct': 0, 'answered': 0},
        method_b_name: {'total': 0, 'correct': 0, 'answered': 0},
    }
    type_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

    pred_map_a = index_predictions_by_id(method_a['predictions'])
    pred_map_b = index_predictions_by_id(method_b['predictions'])

    for data_id in sorted(qid2example.keys()):
        example = qid2example[data_id]
        question_type = str(example.get('question_type', '')).lower()
        pred_a_entry = pred_map_a.get(data_id)
        pred_b_entry = pred_map_b.get(data_id)

        score_a, processed_a = score_method_prediction(example, pred_a_entry)
        score_b, processed_b = score_method_prediction(example, pred_b_entry)

        method_view_a = build_method_view(
            pred_a_entry,
            score_a,
            processed_a,
            include_selected_title=False,
            include_selected_url=False,
        )
        method_view_b = build_method_view(
            pred_b_entry,
            score_b,
            processed_b,
            include_selected_title=True,
            include_selected_url=True,
        )

        if method_view_a['correct'] and method_view_b['correct']:
            bucket_key = 'both_correct'
        elif method_view_a['correct'] and not method_view_b['correct']:
            bucket_key = method_a_only_key
        elif not method_view_a['correct'] and method_view_b['correct']:
            bucket_key = method_b_only_key
        else:
            bucket_key = 'both_incorrect'

        data_split = example.get('data_split', '')
        entity_text = example.get('entity_text')
        title_match = title_matches_entity(
            method_view_b.get('selected_title'),
            entity_text,
        )
        if title_match is True:
            title_detail = 'ours_title_match'
        elif title_match is False:
            title_detail = 'ours_title_mismatch'
        else:
            title_detail = 'ours_title_unknown'

        extended_case_label = (
            f'{bucket_key}|split:{data_split}|type:{question_type}|{title_detail}'
            if data_split
            else f'{bucket_key}|type:{question_type}|{title_detail}'
        )

        analysis_entry: Dict[str, Any] = {
            'data_id': data_id,
            'ground_truth_entity': entity_text,
            'question': example.get('question'),
            'question_type': example.get('question_type'),
            'data_split': example.get('data_split'),
            'image_id': example.get('image_id'),
            'ground_truth': example.get('answer'),
            'ground_truth_eval': example.get('answer_eval'),
            'case_label': extended_case_label,
            method_a_name: method_view_a,
            method_b_name: method_view_b,
            'ours_selected_title_match': title_match,
        }
        entity_id = example.get('entity_id')
        if entity_id is not None:
            analysis_entry['entity_id'] = entity_id
        general_buckets[bucket_key].append(analysis_entry)
        detail_label = extended_case_label
        detailed_buckets.setdefault(detail_label, [])
        detailed_buckets[detail_label].append(analysis_entry)

        question_type = str(example.get('question_type', '')).lower()
        for method_name, method_view in [
            (method_a_name, method_view_a),
            (method_b_name, method_view_b),
        ]:
            stats_entry = method_stats[method_name]
            stats_entry['total'] += 1
            if method_view['available']:
                stats_entry['answered'] += 1
            if method_view['correct']:
                stats_entry['correct'] += 1

            qtype_stats = type_stats.setdefault(question_type, {})
            qtype_entry = qtype_stats.setdefault(
                method_name,
                {'total': 0, 'correct': 0, 'answered': 0},
            )
            qtype_entry['total'] += 1
            if method_view['available']:
                qtype_entry['answered'] += 1
            if method_view['correct']:
                qtype_entry['correct'] += 1

    title_match_counts = {'match': 0, 'mismatch': 0, 'unknown': 0}
    for bucket_entries in general_buckets.values():
        for entry in bucket_entries:
            match_status = entry.get('ours_selected_title_match')
            if match_status is True:
                title_match_counts['match'] += 1
            elif match_status is False:
                title_match_counts['mismatch'] += 1
            else:
                title_match_counts['unknown'] += 1

    general_file_map: Dict[str, str] = {}
    for bucket_key in general_buckets:
        path = os.path.join(analysis_dir, f'{pair_slug}_{bucket_key}.jsonl')
        general_file_map[bucket_key] = path
        write_jsonl(path, general_buckets[bucket_key])

    detail_dir = os.path.join(analysis_dir, 'case_labels')
    os.makedirs(detail_dir, exist_ok=True)
    detailed_file_map: Dict[str, str] = {}
    for detail_label, entries in detailed_buckets.items():
        detail_slug = dynamic_bucket_key(detail_label)
        path = os.path.join(detail_dir, f'{pair_slug}_{detail_slug}.jsonl')
        detailed_file_map[detail_label] = path
        write_jsonl(path, entries)

    unique_example_ids = {
        entry['data_id']
        for entries in general_buckets.values()
        for entry in entries
    }

    summary: Dict[str, Any] = {
        'methods': [method_a_name, method_b_name],
        'total_examples': len(unique_example_ids),
        'combination_counts': {
            'general': {
                key: len(entries) for key, entries in general_buckets.items()
            },
            'detailed': {
                key: len(entries) for key, entries in detailed_buckets.items()
            },
        },
        'method_accuracy': {},
        'per_question_type': {},
        'case_files': {
            'general': general_file_map,
            'detailed': detailed_file_map,
        },
        'ours_selected_title_match_counts': title_match_counts,
    }

    for method_name, stats_entry in method_stats.items():
        summary['method_accuracy'][method_name] = {
            'total': stats_entry['total'],
            'answered': stats_entry['answered'],
            'correct': stats_entry['correct'],
            'accuracy_overall': round(
                safe_division(stats_entry['correct'], stats_entry['total']) * 100,
                2,
            ) if stats_entry['total'] else 0.0,
            'accuracy_when_answered': round(
                safe_division(stats_entry['correct'], stats_entry['answered']) * 100,
                2,
            ) if stats_entry['answered'] else 0.0,
        }

    for question_type, qtype_data in type_stats.items():
        summary['per_question_type'][question_type] = {}
        for method_name, stats_entry in qtype_data.items():
            summary['per_question_type'][question_type][method_name] = {
                'total': stats_entry['total'],
                'answered': stats_entry['answered'],
                'correct': stats_entry['correct'],
                'accuracy_overall': round(
                    safe_division(
                        stats_entry['correct'], stats_entry['total']
                    ) * 100,
                    2,
                ) if stats_entry['total'] else 0.0,
                'accuracy_when_answered': round(
                    safe_division(
                        stats_entry['correct'], stats_entry['answered']
                    ) * 100,
                    2,
                ) if stats_entry['answered'] else 0.0,
            }

    summary_path = os.path.join(analysis_dir, f'{pair_slug}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'\nCase analysis saved to: {analysis_dir}')
    for label, path in sorted(general_file_map.items()):
        print(f'  {label}: {len(general_buckets[label])} examples -> {path}')
    if detailed_file_map:
        print('  detailed case labels:')
        for label, path in sorted(detailed_file_map.items()):
            print(f'    {label}: {len(detailed_buckets[label])} examples -> {path}')
    print(f'  summary: {summary_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, nargs='+', required=True)
    parser.add_argument(
        '--method_names',
        type=str,
        nargs='+',
        help='Optional human-readable names for each method (same order as --input_path).',
    )
    parser.add_argument('--adjust_score', action='store_true')
    parser.add_argument(
        '--split',
        type=str,
        choices=['val', 'test'],
        default='val',
        help='Reserved for compatibility; not used directly in enhanced analysis.',
    )
    parser.add_argument(
        '--reference_path',
        type=str,
        default="/data/qianMa/EchoSight/InfoSeek/infoseek_val.jsonl",
    )
    parser.add_argument(
        '--reference_qtype_path',
        type=str,
        default="/data/qianMa/EchoSight/infoseek_val_qtype.jsonl",
    )
    parser.add_argument(
        '--reference_withkb_path',
        type=str,
        default="/data/qianMa/EchoSight/InfoSeek/infoseek_val_withkb.jsonl",
        help='Optional path to the metadata file that provides entity_text/entity_id.',
    )
    parser.add_argument(
        '--analysis_output_dir',
        type=str,
        default='analysis_outputs',
        help='Directory where the detailed case analysis files will be stored.',
    )
    parser.add_argument(
        '--show_eval_progress',
        action='store_true',
        help='Display tqdm progress bars while scoring predictions.',
    )
    args = parser.parse_args()

    if args.method_names:
        if len(args.method_names) != len(args.input_path):
            raise ValueError(
                'The number of --method_names entries must equal the number of '
                '--input_path directories.'
            )
        method_names = args.method_names
    else:
        method_names = [
            os.path.basename(os.path.normpath(path)) or f'method_{idx + 1}'
            for idx, path in enumerate(args.input_path)
        ]

    reference = load_jsonl(args.reference_path)
    reference_qtype = load_jsonl(args.reference_qtype_path)
    base_qid2example = prepare_qid2example(reference, reference_qtype)
    entity_metadata = load_entity_metadata(args.reference_withkb_path)
    for data_id, meta in entity_metadata.items():
        if data_id in base_qid2example:
            if meta.get('entity_text') is not None:
                base_qid2example[data_id]['entity_text'] = meta.get('entity_text')
            if meta.get('entity_id') is not None:
                base_qid2example[data_id]['entity_id'] = meta.get('entity_id')

    methods_data: List[Dict[str, Any]] = []

    for pred_path, method_name in zip(args.input_path, method_names):
        print(f"=== {method_name} ({pred_path}) ===")

        merge_path = ensure_merge_file(pred_path)
        predictions = load_jsonl(merge_path)
        prediction_ids = {
            pred['data_id'] for pred in predictions if 'data_id' in pred
        }

        if args.adjust_score:
            qid2example_subset = filter_qid2example_by_ids(
                base_qid2example, prediction_ids
            )
        else:
            qid2example_subset = base_qid2example
        print(f'Total predictions loaded: {len(predictions)}')
        result = evaluate_predictions_with_qid(
            predictions,
            qid2example_subset,
            show_progress=args.show_eval_progress,
        )
        final_score = result["final_score"]
        unseen_question_score = result["unseen_question_score"]["score"]
        unseen_entity_score = result["unseen_entity_score"]["score"]

        print(
            f"FINAL SCORE: {final_score}\n"
            f"UNSEEN QUESTION SCORE: {unseen_question_score}\n"
            f"unseen question score time: {result['unseen_question_score']['score_time']}\n"
            f"unseen question score num: {result['unseen_question_score']['score_num']}\n"
            f"unseen question score string: {result['unseen_question_score']['score_string']}\n"
            f"UNSEEN ENTITY SCORE: {unseen_entity_score}\n"
            f"unseen entity score time: {result['unseen_entity_score']['score_time']}\n"
            f"unseen entity score num: {result['unseen_entity_score']['score_num']}\n"
            f"unseen entity score string: {result['unseen_entity_score']['score_string']}\n"
        )

        meta_result = {
            'FINAL SCORE': final_score,
            'UNSEEN QUESTION': unseen_question_score,
            'unseen question score time': result['unseen_question_score']['score_time'],
            'unseen question score num': result['unseen_question_score']['score_num'],
            'unseen question score string': result['unseen_question_score']['score_string'],
            'UNSEEN ENTITY SCORE': unseen_entity_score,
            'unseen entity score time': result['unseen_entity_score']['score_time'],
            'unseen entity score num': result['unseen_entity_score']['score_num'],
            'unseen entity score string': result['unseen_entity_score']['score_string'],
        }
        if pred_path.endswith('.jsonl'):
            pred_path_dir = os.path.dirname(pred_path)
        else:
            pred_path_dir = pred_path

        with open(os.path.join(pred_path_dir, 'results.json'), 'w', encoding='utf-8') as f:
            ujson.dump(meta_result, f)

        methods_data.append(
            {
                'name': method_name,
                'path': pred_path,
                'merge_path': merge_path,
                'predictions': predictions,
                'prediction_ids': prediction_ids,
                'evaluation': result,
            }
        )

    if len(methods_data) >= 2:
        reflect_method: Optional[Dict[str, Any]] = None
        ours_method: Optional[Dict[str, Any]] = None
        for method_entry in methods_data:
            name_lower = method_entry['name'].lower()
            if reflect_method is None and 'reflect' in name_lower:
                reflect_method = method_entry
            if ours_method is None and 'ours' in name_lower:
                ours_method = method_entry

        if reflect_method and ours_method and reflect_method is not ours_method:
            method_a = reflect_method
            method_b = ours_method
        else:
            method_a = methods_data[0]
            method_b = methods_data[1]

        if args.adjust_score:
            analysis_ids = method_a['prediction_ids'].union(method_b['prediction_ids'])
            analysis_qid2example = filter_qid2example_by_ids(
                base_qid2example, analysis_ids
            )
        else:
            analysis_qid2example = base_qid2example

        output_root = (
            args.analysis_output_dir
            if os.path.isabs(args.analysis_output_dir)
            else os.path.join(os.getcwd(), args.analysis_output_dir)
        )
        os.makedirs(output_root, exist_ok=True)

        run_case_analysis(method_a, method_b, analysis_qid2example, output_root)
        if len(methods_data) > 2:
            print(
                "\nNote: case analysis is currently generated for the first two "
                "methods only."
            )
