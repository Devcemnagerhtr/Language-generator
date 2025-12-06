import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gruut_ipa import Phoneme

# =========================
#  PHONEME INVENTORY
# =========================

LIBRARY = [
    'p', 'b', 't', 'd', 'k', 'g',
    'm', 'n',
    'f', 'v', 's', 'h',
    'l', 'ɾ', 'j', 'w',
    'i', 'e', 'a', 'o', 'u'
]

CONSONANTS = ['p', 'b', 't', 'd', 'k', 'g',
              'm', 'n',
              'f', 'v', 's', 'h',
              'l', 'ɾ', 'j', 'w']

VOWELS = ['i', 'e', 'a', 'o', 'u']

# =========================
#  HYPERPARAMETERS
# =========================

LR = 0.005        # slightly smaller LR for larger dataset
EPOCHS = 2000      # increase this if you want very long training
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CV-structure bias hyperparameters
CV_BONUS = 2.0    # reward C -> V
VC_BONUS = 0.4    # reward V -> C (for CVC, CVCV)
CC_PENALTY = 4.0  # strong penalty C -> C
VV_PENALTY = 0.8  # penalty V -> V (diphthongs allowed but rarer)

# Length prior for greetings (in phonemes)
TARGET_LEN = 5        # typical greeting ~5 segments (≈ 2–3 syllables)
LENGTH_SIGMA = 1.5    # how sharply we prefer that length

# =========================
#  FEATURE MAPS
# =========================

OPENNESS_MAP = {
    "plosive": 0.0,
    "nasal": 0.1,
    "fricative": 0.2,
    "approximant": 0.4,
    "close": 0.6,
    "near-close": 0.7,
    "close-mid": 0.8,
    "mid": 0.85,
    "open-mid": 0.9,
    "near-open": 0.95,
    "open": 1.0,
}

PLACE_MAP = {
    "bilabial": 0.0,
    "labiodental": 0.1,
    "dental": 0.2,
    "alveolar": 0.3,
    "front": 0.3,
    "postalveolar": 0.4,
    "retroflex": 0.5,
    "palatal": 0.6,
    "central": 0.5,
    "velar": 0.8,
    "uvular": 0.9,
    "glottal": 1.0,
    "back": 0.8,
}

# Simple psychoacoustic features: (brightness, noisiness), all scaled 0–1
ACOUSTIC_MAP = {
    'p': (0.4, 0.4),
    'b': (0.3, 0.4),
    't': (0.6, 0.4),
    'd': (0.5, 0.4),
    'k': (0.5, 0.4),
    'g': (0.4, 0.4),
    'm': (0.2, 0.2),
    'n': (0.3, 0.2),
    'f': (0.5, 0.8),
    'v': (0.5, 0.7),
    's': (0.8, 0.9),
    'h': (0.5, 0.7),
    'l': (0.4, 0.3),
    'ɾ': (0.5, 0.3),
    'j': (0.7, 0.3),
    'w': (0.3, 0.3),
    'i': (0.9, 0.1),
    'e': (0.7, 0.1),
    'a': (0.6, 0.1),
    'o': (0.4, 0.1),
    'u': (0.3, 0.1),
}

# =========================
#  PHONEME → VECTOR
# =========================

def get_phoneme_vector(ch):
    try:
        p = Phoneme(ch)
    except Exception:
        return None

    x_place = 0.5
    y_open = 0.5
    is_vowel = 0.0

    if p.vowel:
        is_vowel = 1.0

        height = getattr(p.vowel, "height", None)
        if height is not None:
            key = getattr(height, "value", height)
            y_open = OPENNESS_MAP.get(key, y_open)

        placement = getattr(p.vowel, "placement", None)
        if placement is not None:
            key = getattr(placement, "value", placement)
            x_place = PLACE_MAP.get(key, x_place)

    elif p.consonant:
        is_vowel = 0.0

        place = getattr(p.consonant, "place", None)
        if place is not None:
            key = getattr(place, "value", place)
            x_place = PLACE_MAP.get(key, x_place)

        c_type = getattr(p.consonant, "type", None)
        if c_type is not None:
            key = getattr(c_type, "value", c_type)
            y_open = OPENNESS_MAP.get(key, y_open)

    # Articulatory part
    base_vec = np.array([x_place, y_open, is_vowel], dtype=np.float32)

    # Acoustic part (brightness, noisiness); default if missing
    brightness, noisiness = ACOUSTIC_MAP.get(ch, (0.5, 0.3))
    acous_vec = np.array([brightness, noisiness], dtype=np.float32)

    # Concatenate into one feature vector
    return np.concatenate([base_vec, acous_vec])

# =========================
#  NEURAL MODEL
# =========================

class BasicPhonotacticNN(nn.Module):
    def __init__(self, phoneme_vectors):
        super().__init__()
        self.phoneme_vectors = phoneme_vectors
        V, D = phoneme_vectors.shape
        self.V = V
        self.D = D

        # φ(i,j) = [v_i, v_j, |v_j - v_i|] ∈ R^{3D}
        self.global_W = nn.Linear(3 * D, 1, bias=True)
        self.phone_W = nn.Parameter(torch.zeros(V, D))
        self.bigram_B = nn.Parameter(torch.zeros(V, V))

    def forward(self, prev_idx):
        batch_size = prev_idx.shape[0]
        V, D = self.V, self.D

        v_i = self.phoneme_vectors[prev_idx]   # (batch, D)
        v_j = self.phoneme_vectors            # (V, D)

        v_i_exp = v_i.unsqueeze(1).expand(batch_size, V, D)  # (batch, V, D)
        v_j_exp = v_j.unsqueeze(0).expand(batch_size, V, D)  # (batch, V, D)
        diff = torch.abs(v_j_exp - v_i_exp)                  # (batch, V, D)

        feat = torch.cat([v_i_exp, v_j_exp, diff], dim=-1)   # (batch, V, 3D)
        base_scores = self.global_W(feat).squeeze(-1)        # (batch, V)

        phone_term = (self.phone_W * self.phoneme_vectors).sum(dim=-1)  # (V,)
        phone_term = phone_term.unsqueeze(0).expand(batch_size, V)      # (batch, V)

        bigram_term = self.bigram_B[prev_idx, :]                         # (batch, V)

        scores = base_scores + phone_term + bigram_term                  # (batch, V)

        # ----- CV / VC / CC / VV structural bias -----
        # is_vowel is the 3rd component of each phoneme vector
        is_vowel_all = self.phoneme_vectors[:, 2]  # (V,)

        prev_is_vowel = is_vowel_all[prev_idx]     # (batch,)
        prev_is_vowel_exp = prev_is_vowel.unsqueeze(1).expand(batch_size, V)
        next_is_vowel_exp = is_vowel_all.unsqueeze(0).expand(batch_size, V)

        is_C_prev = (prev_is_vowel_exp == 0)
        is_V_prev = (prev_is_vowel_exp == 1)
        is_C_next = (next_is_vowel_exp == 0)
        is_V_next = (next_is_vowel_exp == 1)

        cv_mask = is_C_prev & is_V_next   # C -> V
        vc_mask = is_V_prev & is_C_next   # V -> C
        cc_mask = is_C_prev & is_C_next   # C -> C
        vv_mask = is_V_prev & is_V_next   # V -> V

        struct_term = (
            CV_BONUS * cv_mask.float()
            + VC_BONUS * vc_mask.float()
            - CC_PENALTY * cc_mask.float()
            - VV_PENALTY * vv_mask.float()
        )

        scores = scores + struct_term

        return scores

# =========================
#  TRAINING DATA
# =========================

def build_training_words():
    """
    Build a large synthetic corpus of pleasant, greeting-like words.
    Mostly CV, CVC, and CVCV patterns, overweighting 'friendly' consonants.
    """
    words = []

    # 1. Single-syllable CV words
    for c in CONSONANTS:
        for v in VOWELS:
            words.append(c + v)

    # 2. CVC words
    for c1 in CONSONANTS:
        for v in VOWELS:
            for c2 in CONSONANTS:
                words.append(c1 + v + c2)

    # 3. Two-syllable CV-CV words
    for c1 in CONSONANTS:
        for v1 in VOWELS:
            for c2 in CONSONANTS:
                for v2 in VOWELS:
                    words.append(c1 + v1 + c2 + v2)

    # 4. Overweight some "greeting-ish" patterns
    greeting_like = [
        "mama", "mama", "mama",   # repeated to boost frequency
        "nana", "nana",
        "sala", "sala",
        "hala", "hawa", "hano",
        "kala", "kana", "kawa",
        "lami", "sami", "nami",
        "wena", "wena", "wena",
        "salo", "halo", "halo",
        "mano", "mino", "mio",
    ]
    words.extend(greeting_like * 20)  # add many copies

    # Shuffle so batches get a mix
    random.shuffle(words)
    return words

def words_to_transitions(words, char2idx):
    pairs = []
    for w in words:
        phones = [ch for ch in w if ch in char2idx]
        for i in range(len(phones) - 1):
            prev_ch = phones[i]
            next_ch = phones[i + 1]
            pairs.append((char2idx[prev_ch], char2idx[next_ch]))
    return pairs

# =========================
#  TRAINING LOOP
# =========================

def train_epoch(model, optimizer, criterion, prev_indices, next_indices, batch_size=32):
    model.train()
    N = prev_indices.shape[0]
    perm = torch.randperm(N, device=prev_indices.device)
    total_loss = 0.0

    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        batch_prev = prev_indices[idx]
        batch_next = next_indices[idx]

        optimizer.zero_grad()
        scores = model(batch_prev)
        loss = criterion(scores, batch_next)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch_prev.shape[0]

    return total_loss / N

# =========================
#  GENERATION
# =========================

def generate_word(model, idx2char, length=6, start_char=None):
    """
    Generation with a hard CV constraint:
      - start on a consonant (unless start_char overrides)
      - after a consonant, force a vowel (except possibly at the last position)
    """
    model.eval()
    V = len(idx2char)

    is_vowel_all = (model.phoneme_vectors[:, 2] > 0.5)  # (V,)
    consonant_indices = (~is_vowel_all).nonzero(as_tuple=False).squeeze(1)
    vowel_indices = is_vowel_all.nonzero(as_tuple=False).squeeze(1)

    if start_char is None:
        # start with a consonant
        start_pos = torch.randint(len(consonant_indices), (1,)).item()
        current_idx = int(consonant_indices[start_pos])
    else:
        current_idx = char2idx[start_char]

    word = [idx2char[current_idx]]

    for pos in range(1, length):
        prev_tensor = torch.tensor([current_idx], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            scores = model(prev_tensor).squeeze(0)  # (V,)

        # If previous is consonant and not at final position,
        # mask all consonants so next must be a vowel.
        if (not bool(is_vowel_all[current_idx])) and (pos < length - 1):
            scores[consonant_indices] = -1e9

        probs = torch.softmax(scores, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        word.append(idx2char[next_idx])
        current_idx = next_idx

    return "".join(word)

def length_prior_score(word_len, target=TARGET_LEN, sigma=LENGTH_SIGMA):
    """
    Gaussian log-prior on length: peak at target length.
    """
    return -((word_len - target) ** 2) / (2 * sigma * sigma)

def generate_greeting(model, idx2char, num_candidates=200):
    """
    Generate many candidate words with random lengths,
    score them with a length prior + average transition log-prob,
    return the best-scoring one.
    """
    model.eval()
    best_word = None
    best_score = -1e9

    V = len(idx2char)
    is_vowel_all = (model.phoneme_vectors[:, 2] > 0.5)
    consonant_indices = (~is_vowel_all).nonzero(as_tuple=False).squeeze(1)

    for _ in range(num_candidates):
        # sample a length between 3 and 8, biased around TARGET_LEN
        base = TARGET_LEN
        jitter = random.randint(-2, 2)
        length = max(3, min(8, base + jitter))

        # generate a word of that length
        word = generate_word(model, idx2char, length=length)

        # compute approximate log-prob under the model for this sequence
        # (ignoring length prior already; we'll add that)
        indices = [char2idx[ch] for ch in word]
        log_prob = 0.0
        for i in range(len(indices) - 1):
            prev_idx = torch.tensor([indices[i]], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                scores = model(prev_idx).squeeze(0)
                probs = torch.softmax(scores, dim=-1)
            log_prob += float(torch.log(probs[indices[i+1]] + 1e-12))

        avg_log_prob = log_prob / max(1, len(indices) - 1)
        lp = length_prior_score(len(indices))

        total_score = avg_log_prob + lp

        if total_score > best_score:
            best_score = total_score
            best_word = word

    return best_word

# =========================
#  MAIN
# =========================

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    char2idx = {ch: i for i, ch in enumerate(LIBRARY)}
    idx2char = {i: ch for ch, i in char2idx.items()}

    vectors = []
    for ch in LIBRARY:
        v = get_phoneme_vector(ch)
        if v is None:
            raise ValueError(f"Could not build vector for IPA symbol '{ch}'")
        vectors.append(v)

    phoneme_vectors = torch.tensor(np.stack(vectors), dtype=torch.float32, device=DEVICE)
    V, D = phoneme_vectors.shape
    print(f"Library size: {V}, feature dim: {D}")

    # build big training corpus
    train_words = build_training_words()
    print(f"Number of training words: {len(train_words)}")

    train_pairs = words_to_transitions(train_words, char2idx)
    if not train_pairs:
        raise RuntimeError("No transitions found. Check that your train_words use symbols from LIBRARY.")

    prev_indices = torch.tensor([p[0] for p in train_pairs], dtype=torch.long, device=DEVICE)
    next_indices = torch.tensor([p[1] for p in train_pairs], dtype=torch.long, device=DEVICE)
    print("Number of transitions:", len(train_pairs))

    model = BasicPhonotacticNN(phoneme_vectors).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        avg_loss = train_epoch(model, optimizer, criterion, prev_indices, next_indices, batch_size=BATCH_SIZE)
        print(f"Epoch {epoch+1:03d} - avg NLL: {avg_loss:.4f}")

    print("\nSample generated words:")
    for i in range(10):
        print(f"{i+1:2d}:", generate_word(model, idx2char, length=5))

    print("\nGreeting candidates:")
    for i in range(10):
        print(f"greet {i+1}:", generate_greeting(model, idx2char, num_candidates=200))
