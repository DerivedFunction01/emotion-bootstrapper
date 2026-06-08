import json
import math
import os
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SEMANTIC_HYPOTHESES = {
    "fear": "Someone has at least at least a strong feeling of fear caused by the threat of danger or pain",
    "anger": "Someone has at least a strong feeling of annoyance, displeasure, or hostility",
    "surprise": "Someone has at least a strong feeling of mild shock or astonishment",
    "joy": "Someone has at least a strong feeling of great pleasure and happiness",
    "sadness": "Someone has at least a strong feeling of deep distress caused by loss or disappointment",
    "disgust": "Someone has at least a strong feeling of revulsion or strong disapproval",
    "urgency": "Someone has at least a strong feeling of urgency to act immediately due to time pressure",
    "confusion": "Someone has at least a strong feeling of struggling to understand something complex or confusing",
    "emotion": "There is someone with feelings of either fear, anger, surprise, joy, sadness, disgust, urgency, or confusion",
}

SEMANTIC_HYPOTHESES_MULTILINGUAL = {
    "english": {
        "fear": "Someone has at least a strong feeling of fear caused by the threat of danger or pain",
        "anger": "Someone has at least a strong feeling of annoyance, displeasure, or hostility",
        "surprise": "Someone has at least a strong feeling of mild shock or astonishment",
        "joy": "Someone has at least a strong feeling of great pleasure and happiness",
        "sadness": "Someone has at least a strong feeling of deep distress caused by loss or disappointment",
        "disgust": "Someone has at least a strong feeling of revulsion or strong disapproval",
        "urgency": "Someone has at least a strong feeling of urgency to act immediately due to time pressure",
        "confusion": "Someone has at least a strong feeling of struggling to understand something complex or confusing",
        "emotion": "There is someone with feelings of either fear, anger, surprise, joy, sadness, disgust, urgency, or confusion",
    },
    "french": {
        "fear": "Quelqu'un ressent au moins un sentiment fort de peur causé par la menace d'un danger ou de la douleur",
        "anger": "Quelqu'un ressent au moins un sentiment fort d'agacement, de mécontentement ou d'hostilité",
        "surprise": "Quelqu'un ressent au moins un sentiment fort de léger choc ou d'étonnement",
        "joy": "Quelqu'un ressent au moins un sentiment fort de grand plaisir et de bonheur",
        "sadness": "Quelqu'un ressent au moins un sentiment fort de détresse profonde causée par une perte ou une déception",
        "disgust": "Quelqu'un ressent au moins un sentiment fort de répulsion ou de profonde désapprobation",
        "urgency": "Quelqu'un ressent au moins un sentiment fort d'urgence à agir immédiatement en raison de la pression du temps",
        "confusion": "Quelqu'un ressent au moins un sentiment fort de difficulté à comprendre quelque chose de complexe ou de déroutant",
        "emotion": "Il y a quelqu'un qui éprouve des sentiments soit de peur, de colère, de surprise, de joie, de tristesse, de dégoût, d'urgence ou de confusion",
    },
    "german": {
        "fear": "Jemand hat zumindest ein starkes Gefühl der Angst, das durch die Androhung von Gefahr oder Schmerz verursacht wird",
        "anger": "Jemand hat zumindest ein starkes Gefühl von Ärger, Unmut oder Feindseligkeit",
        "surprise": "Jemand hat zumindest ein starkes Gefühl eines leichten Schocks oder des Erstaunens",
        "joy": "Jemand hat zumindest ein starkes Gefühl großer Freude und Glückseligkeit",
        "sadness": "Jemand hat zumindest ein starkes Gefühl tiefer Betroffenheit, die durch Verlust oder Enttäuschung verursacht wird",
        "disgust": "Jemand hat zumindest ein starkes Gefühl von Abscheu oder starker Missbilligung",
        "urgency": "Jemand hat zumindest ein starkes Gefühl der Dringlichkeit, aufgrund von Zeitdruck sofort zu handeln",
        "confusion": "Jemand hat zumindest ein starkes Gefühl der Mühe, etwas Komplexes oder Verwirrendes zu verstehen",
        "emotion": "Es gibt jemanden mit Gefühlen von entweder Angst, Wut, Überraschung, Freude, Traurigkeit, Ekel, Dringlichkeit oder Verwirrung",
    },
    "spanish": {
        "fear": "Alguien tiene al menos un fuerte sentimiento de miedo causado por la amenaza de peligro o dolor",
        "anger": "Alguien tiene al menos un fuerte sentimiento de molestia, disgusto u hostilidad",
        "surprise": "Alguien tiene al menos un fuerte sentimiento de leve conmoción o asombro",
        "joy": "Alguien tiene al menos un fuerte sentimiento de gran placer y felicidad",
        "sadness": "Alguien tiene al menos un fuerte sentimiento de profunda angustia causada por una pérdida o decepción",
        "disgust": "Alguien tiene al menos un fuerte sentimiento de repulsión o fuerte desaprobación",
        "urgency": "Alguien tiene al menos un fuerte sentimiento de urgencia por actuar de inmediato debido a la presión del tiempo",
        "confusion": "Alguien tiene al menos un fuerte sentimiento de dificultad para comprender algo complejo o confuso",
        "emotion": "Hay alguien con sentimientos ya sea de miedo, ira, sorpresa, alegría, tristeza, asco, urgencia o confusión",
    },
    "italian": {
        "fear": "Qualcuno prova almeno un forte sentimento di paura causato dalla minaccia di pericolo o dolore",
        "anger": "Qualcuno prova almeno un forte sentimento di fastidio, dispiacere o ostilità",
        "surprise": "Qualcuno prova almeno un forte sentimento di lieve shock o stupore",
        "joy": "Qualcuno prova almeno un forte sentimento di grande piacere e felicità",
        "sadness": "Qualcuno prova almeno un forte sentimento di profonda angoscia causata da una perdita o da una delusione",
        "disgust": "Qualcuno prova almeno un forte sentimento di ripulsione o forte disapprovazione",
        "urgency": "Qualcuno prova almeno un forte sentimento di urgenza di agire immediatamente a causa della pressione del tempo",
        "confusion": "Qualcuno prova almeno un forte sentimento di difficoltà a comprendere qualcosa di complesso o confuso",
        "emotion": "C'è qualcuno con sentimenti di paura, rabbia, sorpresa, gioia, tristezza, disgusto, urgenza o confusione",
    },
    "portuguese": {
        "fear": "Alguém tem pelo menos um forte sentimento de medo causado pela ameaça de perigo ou dor",
        "anger": "Alguém tem pelo menos um forte sentimento de incômodo, desagrado ou hostilidade",
        "surprise": "Alguém tem pelo menos um forte sentimento de leve choque ou espanto",
        "joy": "Alguém tem pelo menos um forte sentimento de grande prazer e felicidade",
        "sadness": "Alguém tem pelo menos um forte sentimento de profunda angústia causada por perda ou decepção",
        "disgust": "Alguém tem pelo menos um forte sentimento de repulsa ou forte desaprovação",
        "urgency": "Alguém tem pelo menos um forte sentimento de urgência para agir imediatamente devido à pressão do tempo",
        "confusion": "Alguém tem pelo menos um forte sentimento de dificuldade para entender algo complexo ou confuso",
        "emotion": "Há alguém com sentimentos de medo, raiva, surpresa, alegria, tristeza, nojo, urgência ou confusão",
    },
    "russian": {
        "fear": "Кто-то испытывает как минимум сильное чувство страха, вызванное угрозой опасности или боли",
        "anger": "Кто-то испытывает как минимум сильное чувство раздражения, недовольства или враждебности",
        "surprise": "Кто-то испытывает как минимум сильное чувство легкого шока или изумления",
        "joy": "Кто-то испытывает как минимум сильное чувство огромного удовольствия и счастья",
        "sadness": "Кто-то испытывает как минимум сильное чувство глубокого огорчения, вызванного утратой или разочарованием",
        "disgust": "Кто-то испытывает как минимум сильное чувство отвращения или сильного неодобрения",
        "urgency": "Кто-то испытывает как минимум сильное чувство необходимости действовать немедленно из-за дефицита времени",
        "confusion": "Кто-то испытывает как минимум сильное чувство растерянности и с трудом понимает что-то сложное или запутанное",
        "emotion": "Есть кто-то, кто испытывает чувства страха, гнева, удивления, радости, печали, отвращения, срочности или замешательства",
    },
    "chinese": {
        "fear": "某人因危险或痛苦的的威胁而产生至少是强烈的恐惧感",
        "anger": "某人产生至少是强烈的恼怒、不满或敌意",
        "surprise": "某人产生至少是强烈的轻微震惊或惊讶感",
        "joy": "某人产生至少是强烈的巨大愉悦和幸福感",
        "sadness": "某人因失去或失望而产生至少是强烈的深度悲伤或痛苦",
        "disgust": "某人产生至少是强烈的反感或强烈反对的情绪",
        "urgency": "某人因时间紧迫而产生至少是强烈的立即行动的紧迫感",
        "confusion": "某人产生至少是强烈的难以理解复杂或令人困惑之事的挣扎感",
        "emotion": "有人正处于恐惧、愤怒、惊讶、喜悦、悲伤、厌恶、紧急或困惑其中一种情感之中",
    },
    "japanese": {
        "fear": "誰かが、危険や痛みの脅威によって引き起こされる少なくとも強い恐怖心抱いている",
        "anger": "誰かが、少なくとも強い苛立ち、不快感、または敵意を抱いている",
        "surprise": "誰かが、少なくとも強い軽いショックや驚きを感じている",
        "joy": "誰かが、少なくとも強い大きな喜びや幸福感を抱いている",
        "sadness": "誰かが、喪失や失望によって引き起こされる少なくとも強い深い悲しみを感じている",
        "disgust": "誰かが、少なくとも強い嫌悪感や強い不承認の念を抱いている",
        "urgency": "誰かが、時間的なプレッシャーにより直ちに行動しなければならないという少なくとも強い切迫感を抱いている",
        "confusion": "誰かが、複雑で紛らわしい何かを理解しようと苦しむ少なくとも強い困惑を抱いている",
        "emotion": "恐怖、怒り、驚き、喜び、悲しみ、嫌悪、切迫、困惑のいずれかの感情を抱いている人がいる",
    },
    "korean": {
        "fear": "누군가가 위험이나 고통의 위협으로 인해 발생하는 최소한 강한 두려움을 느끼고 있다",
        "anger": "누군가가 최소한 강한 짜증, 불쾌감 또는 적대감을 느끼고 있다",
        "surprise": "누군가가 최소한 강한 가벼운 충격이나 놀라움을 느끼고 있다",
        "joy": "누군가가 최소한 강한 큰 기쁨과 행복을 느끼고 있다",
        "sadness": "누군가가 상실이나 실망으로 인해 발생하는 최소한 강한 깊은 슬픔을 느끼고 있다",
        "disgust": "누군가가 최소한 강한 혐오감이나 강한 거부감을 느끼고 있다",
        "urgency": "누군가가 시간적 압박으로 인해 즉각 행동해야 한다는 최소한 강한 긴박감을 느끼고 있다",
        "confusion": "누군가가 복잡하거나 혼란스러운 것을 이해하기 위해 애쓰는 최소한 강한 혼란을 느끼고 있다",
        "emotion": "두려움, 분노, 놀람, 기쁨, 슬픔, 혐오, 긴박, 혼란 중 하나의 감정을 느끼는 누군가가 있다",
    },
    "arabic": {
        "fear": "شخص ما لديه على الأقل شعور قوي بالخوف الناتج عن التهديد بالخطر أو الألم",
        "anger": "شخص ما لديه على الأقل شعور قوي بالانزعاج أو الاستياء أو العداء",
        "surprise": "شخص ما لديه على الأقل شعور قوي بصدمة خفيفة أو دهشة",
        "joy": "شخص ما لديه على الأقل شعور قوي بالمتعة الكبيرة والسعادة",
        "sadness": "شخص ما لديه على الأقل شعور قوي بأسى عميق ناتج عن خسارة أو خيبة أمل",
        "disgust": "شخص ما لديه على الأقل شعور قوي بالاشمئزاز أو الرفض الشديد",
        "urgency": "شخص ما لديه على الأقل شعور قوي بالإلحاح للتحرك فوراً بسبب ضيق الوقت",
        "confusion": "شخص ما لديه على الأقل شعور قوي بالمعاناة لفهم شيء معقد أو مربك",
        "emotion": "هناك شخص لديه مشاعر إما الخوف، أو الغضب، أو المفاجأة، أو الفرح، أو الحزن، أو الاشمئزاز، أو الاستعجال، أو الحيرة",
    },
    "danish": {
        "fear": "Nogen har i det mindste en stærk følelse af frygt forårsaget af truslen om fare eller smerte",
        "anger": "Nogen har i det mindste en stærk følelse af irritation, utilfredshed eller fjendtlighed",
        "surprise": "Nogen har i det mindste en stærk følelse af mildt chok eller forundring",
        "joy": "Nogen har i det mindste en stærk følelse af stor glæde og lykke",
        "sadness": "Nogen har i det mindste en stærk følelse af dyb sorg forårsaget af tab eller skuffelse",
        "disgust": "Nogen har i det mindste en stærk følelse af afsky eller stærk misbilligelse",
        "urgency": "Nogen har i det mindste en stærk følelse af, at det haster med at handle med det samme på grund af tidspres",
        "confusion": "Nogen har i det mindste en stærk følelse af at kæmpe med at forstå noget komplekst eller forvirrende",
        "emotion": "Der er nogen med følelser af enten frygt, vrede, overraskelse, glæde, sorg, afsky, hastværk eller forvirring",
    },
    "polish": {
        "fear": "Ktoś ma co najmniej silne poczucie strachu wywołane groźbą niebezpieczeństwa lub bólu",
        "anger": "Ktoś ma co najmniej silne poczucie irytacji, niezadowolenia lub wrogości",
        "surprise": "Ktoś ma co najmniej silne poczucie lekkiego szoku lub zadziwienia",
        "joy": "Ktoś ma co najmniej silne poczucie wielkiej przyjemności i szczęścia",
        "sadness": "Ktoś ma co najmniej silne poczucie głębokiego żalu wywołanego stratą lub rozczarowaniem",
        "disgust": "Ktoś ma co najmniej silne poczucie wstrętu lub silnej dezaprobaty",
        "urgency": "Ktoś ma co najmniej silne poczucie pilności, by działać natychmiast z powodu presji czasu",
        "confusion": "Ktoś ma co najmniej silne poczucie zagubienia i trudności w zrozumieniu czegoś złożonego lub mętnego",
        "emotion": "Jest ktoś, kto żywi uczucia strachu, gniewu, zaskoczenia, radości, smutku, wstrętu, pilności lub zagubienia",
    },
    "hindi": {
        "fear": "किसी को खतरे या दर्द के डर से कम से कम एक मजबूत डर की भावना महसूस हो रही है",
        "anger": "किसी को कम से कम झुंझलाहट, नाराजगी या शत्रुता की एक मजबूत भावना महसूस हो रही है",
        "surprise": "किसी को कम से कम हल्के झटके या आश्चर्य की एक मजबूत भावना महसूस हो रही है",
        "joy": "किसी को कम से कम बड़े आनंद और खुशी की एक मजबूत भावना महसूस हो रही है",
        "sadness": "किसी को नुकसान या निराशा के कारण कम से कम गहरे दुख की एक मजबूत भावना महसूस हो रही है",
        "disgust": "किसी को कम से कम घृणा या तीव्र अस्वीकृति की एक मजबूत भावना महसूस हो रही है", 
        "urgency": "समय के दबाव के कारण किसी को तुरंत कार्रवाई करने की कम से कम एक मजबूत तात्कालिकता की भावना महसूस हो रही है",
        "confusion": "किसी को किसी जटिल या उलझाने वाली चीज़ को समझने में कम से कम एक मजबूत असमंजस की भावना महसूस हो रही है",
        "emotion": "कोई ऐसा व्यक्ति है जिसे या तो डर, गुस्सा, आश्चर्य, खुशी, दुख, घृणा, तात्कालिकता या असमंजस की भावनाएं महसूस हो रही हैं",
    },
    "indonesian": {
        "fear": "Seseorang setidaknya memiliki rasa takut yang kuat yang disebabkan oleh ancaman bahaya atau rasa sakit",
        "anger": "Seseorang setidaknya memiliki rasa jengkel, tidak senang, atau permusuhan yang kuat",
        "surprise": "Seseorang setidaknya memiliki rasa terkejut atau heran yang kuat",
        "joy": "Seseorang setidaknya memiliki rasa senang dan bahagia yang kuat",
        "sadness": "Seseorang setidaknya memiliki rasa duka yang mendalam yang disebabkan oleh kehilangan atau kekecewaan",
        "disgust": "Seseorang setidaknya memiliki rasa muak atau ketidaksetujuan yang kuat",
        "urgency": "Seseorang setidaknya memiliki rasa mendesak yang kuat untuk segera bertindak karena tekanan waktu",
        "confusion": "Seseorang setidaknya memiliki rasa bingung yang kuat untuk memahami sesuatu yang kompleks atau membingungkan",
        "emotion": "Ada seseorang yang memiliki salah satu perasaan dari rasa takut, marah, terkejut, gembira, sedih, muak, mendesak, atau bingung",
    },
    "turkish": {
        "fear": "Bir kimse, en azından bir tehlike veya acı tehdidinin neden olduğu güçlü bir korku duygusuna sahip",
        "anger": "Bir kimse, en azından güçlü bir rahatsızlık, hoşnutsuzluk veya düşmanlık duygusuna sahip",
        "surprise": "Bir kimse, en azından güçlü bir hafif şok veya şaşkınlık duygusuna sahip",
        "joy": "Bir kimse, en azından güçlü bir büyük memnuniyet ve mutluluk duygusuna sahip",
        "sadness": "Bir kimse, en azından bir kayıp veya hayal kırıklığının neden olduğu güçlü bir derin üzüntü duygusuna sahip",
        "disgust": "Bir kimse, en azından güçlü bir iğrenme veya güçlü bir onaylamama duygusuna sahip",
        "urgency": "Bir kimse, zaman baskısı nedeniyle en azından güçlü bir acilen harekete geçme duygusuna sahip",
        "confusion": "Bir kimse, karmaşık veya kafa karıştırıcı bir şeyi anlamakta en azından güçlü bir zorlanma duygusuna sahip",
        "emotion": "Korku, öfke, sürpriz, sevinç, üzüntü, tiksinti, aciliyet veya kafa karışıklığı duygularından birine sahip biri var",
    },
    "vietnamese": {
        "fear": "Ai đó có ít nhất một cảm giác sợ hãi mãnh liệt do mối đe dọa từ nguy hiểm hoặc đau đớn gây ra",
        "anger": "Ai đó có ít nhất một cảm giác bực bội, khó chịu hoặc thù địch mạnh mẽ",
        "surprise": "Ai đó có ít nhất một cảm giác sốc nhẹ hoặc ngạc nhiên mạnh mẽ",
        "joy": "Ai đó có ít nhất một cảm giác vui sướng và hạnh phúc tràn ngập mạnh mẽ",
        "sadness": "Ai đó có ít nhất một cảm giác đau buồn sâu sắc mạnh mẽ do mất mát hoặc thất vọng gây ra",
        "disgust": "Ai đó có ít nhất một cảm giác ghê tởm hoặc phản đối mạnh mẽ",
        "urgency": "Ai đó có ít nhất một cảm giác cấp bách mạnh mẽ cần phải hành động ngay lập tức do áp lực thời gian",
        "confusion": "Ai đó có ít nhất một cảm giác bối rối mạnh mẽ khi cố gắng hiểu một điều gì đó phức tạp hoặc khó hiểu",
        "emotion": "Có ai đó đang có cảm xúc sợ hãi, tức giận, ngạc nhiên, vui sướng, buồn bã, ghê tởm, cấp bách hoặc bối rối",
    },
}

def load_primary_dataset(dataset_path: str, dataset_config: str) -> Dataset:
    """Load the most relevant split from a Hugging Face dataset."""
    dataset_dict = load_dataset(dataset_path, dataset_config)

    if "train" in dataset_dict:
        return dataset_dict["train"]

    first_split_name = next(iter(dataset_dict.keys()))
    return dataset_dict[first_split_name]


def save_dataset_as_parquet(dataset: Dataset, output_path: str) -> None:
    """Persist a Hugging Face Dataset to parquet."""
    df = dataset.to_pandas()
    df.to_parquet(output_path, index=False)


def load_parquet_size_mb(output_path: str) -> float:
    """Read parquet back so we can report file size / memory usage."""
    return float(pd.read_parquet(output_path).memory_usage(deep=True).sum() / 1e6)


def write_json_atomic(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def chunked(items: List[str], size: int) -> List[List[str]]:
    """Yield consecutive chunks from a list."""
    if size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return [items[i : i + size] for i in range(0, len(items), size)]


def zip_directory(source_dir: str, zip_path: str) -> str:
    """Zip a directory recursively."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                arcname = os.path.relpath(file_path, source_dir)
                zf.write(file_path, arcname)
    return zip_path


class VerboseSemanticBootstrapper:
    """
    Bootstrap emotion labels using verbose semantic hypotheses.
    Optimized for high-entropy emotional text and negation sensitivity.
    """

    def __init__(
        self, model: str | None = None, device_map: str = "auto", multilingual: bool = False
    ):
        self.multilingual = multilingual
        if model is None:
            model = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" if multilingual else "facebook/bart-large-mnli"
        print(f"Loading model/tokenizer: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device_map != "cpu" else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()
        if self.multilingual:
            self.emotion_labels = list(SEMANTIC_HYPOTHESES_MULTILINGUAL["english"].keys())
            self.hypotheses = list(SEMANTIC_HYPOTHESES_MULTILINGUAL["english"].values())
        else:
            self.emotion_labels = list(SEMANTIC_HYPOTHESES.keys())
            self.hypotheses = list(SEMANTIC_HYPOTHESES.values())
        self.entailment_id = self._find_entailment_id()

    def _find_entailment_id(self) -> int:
        label2id = getattr(self.model.config, "label2id", {}) or {}
        for key, idx in label2id.items():
            if str(key).lower() == "entailment":
                return int(idx)
        return 2

    def label_text(self, text: str) -> Dict[str, float]:
        return self.label_texts([text])[0]

    def _tokenize_pairs(
        self, batch: Dict[str, List[str]], indices: List[int], text_column: str
    ) -> Dict[str, List]:
        texts = batch[text_column]
        if self.multilingual and "translation_language" in batch:
            languages = batch["translation_language"]
        else:
            languages = ["english"] * len(texts)

        input_texts = []
        hypotheses = []
        text_indices = []
        label_indices = []

        for text_index, text in enumerate(texts):
            global_text_index = int(indices[text_index])
            lang = str(languages[text_index]).lower()
            if self.multilingual and lang in SEMANTIC_HYPOTHESES_MULTILINGUAL:
                current_hypotheses = list(SEMANTIC_HYPOTHESES_MULTILINGUAL[lang].values())
            else:
                current_hypotheses = self.hypotheses

            for label_index, hypothesis in enumerate(current_hypotheses):
                input_texts.append(text)
                hypotheses.append(hypothesis)
                text_indices.append(global_text_index)
                label_indices.append(label_index)

        tokenized = self.tokenizer(
            input_texts,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokenized["text_index"] = text_indices
        tokenized["label_index"] = label_indices
        return tokenized

    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        batch_size: int = 1000,
        num_proc: int | None = None,
        cache_dir: str | None = None,
    ) -> Dataset:
        print(
            f"Tokenizing {len(dataset)} rows into premise/hypothesis pairs "
            f"with num_proc={num_proc or 1}..."
        )

        tokenized = dataset.map(
            lambda batch, indices: self._tokenize_pairs(batch, indices, text_column),
            batched=True,
            with_indices=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="tokenizing",
        )
        if cache_dir:
            tokenized.save_to_disk(cache_dir)
        return tokenized

    def label_texts(self, texts: List[str], inference_batch_size: int = 32) -> List[Dict[str, float]]:
        if not texts:
            return []

        pair_texts = []
        pair_hypotheses = []
        for text in texts:
            for hypothesis in self.hypotheses:
                pair_texts.append(text)
                pair_hypotheses.append(hypothesis)

        all_entailment_scores: List[float] = []
        for start in tqdm(
            range(0, len(pair_texts), inference_batch_size),
            total=math.ceil(len(pair_texts) / inference_batch_size),
            desc="inference",
        ):
            end = start + inference_batch_size
            inputs = self.tokenizer(
                pair_texts[start:end],
                pair_hypotheses[start:end],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                all_entailment_scores.extend(
                    probs[:, self.entailment_id].detach().cpu().tolist()
                )

        grouped = [
            all_entailment_scores[i : i + len(self.hypotheses)]
            for i in range(0, len(all_entailment_scores), len(self.hypotheses))
        ]
        return [
            {
                emotion: float(score)
                for emotion, score in zip(self.emotion_labels, scores)
            }
            for scores in grouped
        ]

    def bootstrap_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        batch_size: int = 32,
        show_progress: bool = True,
        num_proc: int | None = None,
        tokenized_cache_path: str | None = None,
        tokenized_zip_path: str | None = None,
        raw_cache_path: str | None = None,
    ) -> Dataset:
        if raw_cache_path:
            os.makedirs(raw_cache_path, exist_ok=True)
            Dataset.from_dict({"text": dataset[text_column]}).to_parquet(
                os.path.join(raw_cache_path, "texts.parquet")
            )
            write_json_atomic(
                os.path.join(raw_cache_path, "cache_meta.json"),
                {
                    "dataset_rows": len(dataset),
                    "text_column": text_column,
                    "num_emotions": len(self.hypotheses),
                    "batch_size": batch_size,
                },
            )

        tokenized_dataset = self.tokenize_dataset(
            dataset,
            text_column=text_column,
            batch_size=batch_size,
            num_proc=num_proc,
            cache_dir=tokenized_cache_path,
        )
        if tokenized_cache_path and tokenized_zip_path:
            if os.path.exists(tokenized_zip_path):
                raise FileExistsError(f"Refusing to overwrite existing zip: {tokenized_zip_path}")
            zip_directory(tokenized_cache_path, tokenized_zip_path)

        text_to_scores: Dict[int, List[float]] = {}
        pair_batch_size = max(1, batch_size * len(self.hypotheses))

        for start in tqdm(
            range(0, len(tokenized_dataset), pair_batch_size),
            total=math.ceil(len(tokenized_dataset) / pair_batch_size),
            disable=not show_progress,
            desc="bootstrapping",
        ):
            batch = tokenized_dataset[start : start + pair_batch_size]
            inputs = {
                key: torch.tensor(value).to(self.device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                entailment_scores = probs[:, self.entailment_id].detach().cpu().tolist()

            for text_index, label_index, score in zip(
                batch["text_index"], batch["label_index"], entailment_scores
            ):
                text_to_scores.setdefault(text_index, [0.0] * len(self.hypotheses))
                text_to_scores[text_index][label_index] = float(score)

        emotion_vectors = [
            {
                emotion: float(score)
                for emotion, score in zip(self.emotion_labels, text_to_scores[i])
            }
            for i in range(len(dataset))
        ]

        return dataset.add_column("emotion_vector", emotion_vectors)

    def get_statistics(self, dataset: Dataset) -> Dict:
        emotion_vectors = dataset["emotion_vector"]

        emotion_stats = {}
        for emotion in self.emotion_labels:
            scores = [ev[emotion] for ev in emotion_vectors]
            emotion_stats[emotion] = {
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "count_above_0.5": sum(1 for s in scores if s > 0.5),
                "count_above_0.7": sum(1 for s in scores if s > 0.7),
            }

        num_emotions_per_text = [
            sum(1 for score in ev.values() if score > 0.5) for ev in emotion_vectors
        ]

        return {
            "total_texts": len(emotion_vectors),
            "emotion_statistics": emotion_stats,
            "multi_emotion_distribution": {
                "mean_emotions_per_text": float(np.mean(num_emotions_per_text)),
                "median_emotions_per_text": float(np.median(num_emotions_per_text)),
                "max_emotions_per_text": int(np.max(num_emotions_per_text)),
                "texts_with_single_emotion": sum(1 for n in num_emotions_per_text if n == 1),
                "texts_with_multiple_emotions": sum(1 for n in num_emotions_per_text if n > 1),
                "texts_with_no_clear_emotion": sum(1 for n in num_emotions_per_text if n == 0),
            },
        }

    def print_statistics(self, stats: Dict) -> None:
        print("\n" + "=" * 80)
        print("BOOTSTRAPPED DATASET STATISTICS")
        print("=" * 80)
        print(f"\nTotal texts: {stats['total_texts']}")

        print("\nEmotion Statistics:")
        print("-" * 80)
        print(
            f"{'Emotion':<12} {'Mean':<8} {'Median':<8} {'Std':<8} {'>0.5':<8} {'>0.7':<8}"
        )
        print("-" * 80)
        for emotion, stats_dict in stats["emotion_statistics"].items():
            print(
                f"{emotion:<12} {stats_dict['mean']:<8.3f} {stats_dict['median']:<8.3f} "
                f"{stats_dict['std']:<8.3f} {stats_dict['count_above_0.5']:<8} "
                f"{stats_dict['count_above_0.7']:<8}"
            )

        print("\nMulti-Emotion Distribution:")
        print("-" * 80)
        me_dist = stats["multi_emotion_distribution"]
        print(f"Average emotions per text: {me_dist['mean_emotions_per_text']:.2f}")
        print(f"Median emotions per text: {me_dist['median_emotions_per_text']:.1f}")
        print(f"Max emotions in single text: {me_dist['max_emotions_per_text']}")
        print(f"Texts with single clear emotion: {me_dist['texts_with_single_emotion']}")
        print(f"Texts with multiple emotions: {me_dist['texts_with_multiple_emotions']}")
        print(f"Texts with no clear emotion: {me_dist['texts_with_no_clear_emotion']}")
        print("=" * 80 + "\n")
