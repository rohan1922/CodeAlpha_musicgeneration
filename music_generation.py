"""
CodeAlpha_MusicGeneration
Music Generation with AI using an LSTM (RNN) trained on MIDI files.

Usage:
    python music_generation.py train      # preprocess + train the model
    python music_generation.py generate   # generate a new MIDI file

Put your .mid files (classical, jazz, etc.) inside the folder: midi_songs/
"""

import glob
import os
import pickle
import sys

import numpy as np
from music21 import chord, converter, instrument, note, stream
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Activation, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

# ---------------- Settings ----------------
MIDI_DIR = "midi_songs"
NOTES_FILE = "data/notes.pkl"
MODEL_FILE = "model.keras"
OUTPUT_FILE = "generated_music.mid"
SEQUENCE_LENGTH = 50   # how many previous notes the model looks at
EPOCHS = 50
BATCH_SIZE = 64
NUM_NOTES_TO_GENERATE = 300
TEMPERATURE = 0.9      # lower = safer/repetitive, higher = more random


# ---------------- 1. Data collection + preprocessing ----------------
def extract_notes():
    """Read every MIDI file and turn it into a flat list of notes/chords."""
    files = glob.glob(os.path.join(MIDI_DIR, "*.mid")) + glob.glob(os.path.join(MIDI_DIR, "*.midi"))
    if not files:
        sys.exit(f"No MIDI files found in '{MIDI_DIR}/'. Add some .mid files first.")

    notes = []
    for i, file in enumerate(files, 1):
        print(f"Parsing {i}/{len(files)}: {file}")
        try:
            midi = converter.parse(file)
        except Exception as e:
            print("  skipped:", e)
            continue

        parts = instrument.partitionByInstrument(midi)
        elements = parts.parts[0].recurse() if parts else midi.flat.notes

        for el in elements:
            if isinstance(el, note.Note):
                notes.append(str(el.pitch))
            elif isinstance(el, chord.Chord):
                notes.append(".".join(str(n) for n in el.normalOrder))

    os.makedirs("data", exist_ok=True)
    with open(NOTES_FILE, "wb") as f:
        pickle.dump(notes, f)
    print(f"Total notes/chords extracted: {len(notes)}")
    return notes


def prepare_sequences(notes, vocab_size):
    """Create (input sequence -> next note) training pairs."""
    pitches = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitches)}

    X, y = [], []
    for i in range(len(notes) - SEQUENCE_LENGTH):
        X.append([note_to_int[n] for n in notes[i:i + SEQUENCE_LENGTH]])
        y.append(note_to_int[notes[i + SEQUENCE_LENGTH]])

    n_patterns = len(X)
    X_norm = np.reshape(X, (n_patterns, SEQUENCE_LENGTH, 1)) / float(vocab_size)
    y_cat = to_categorical(y, num_classes=vocab_size)
    return X, X_norm, y_cat, note_to_int


# ---------------- 2. Model ----------------
def build_model(input_shape, vocab_size):
    model = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(vocab_size),
        Activation("softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


# ---------------- 3. Train ----------------
def train():
    notes = extract_notes()
    vocab_size = len(set(notes))
    _, X, y, _ = prepare_sequences(notes, vocab_size)

    model = build_model((X.shape[1], X.shape[2]), vocab_size)
    model.summary()

    checkpoint = ModelCheckpoint(MODEL_FILE, monitor="loss", save_best_only=True, verbose=1)
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint])
    print("Training done. Model saved to", MODEL_FILE)


# ---------------- 4. Generate ----------------
def sample_with_temperature(probs, temperature):
    probs = np.log(probs + 1e-9) / temperature
    probs = np.exp(probs) / np.sum(np.exp(probs))
    return np.random.choice(len(probs), p=probs)


def generate():
    with open(NOTES_FILE, "rb") as f:
        notes = pickle.load(f)

    vocab_size = len(set(notes))
    X_int, _, _, note_to_int = prepare_sequences(notes, vocab_size)
    int_to_note = {i: n for n, i in note_to_int.items()}
    model = load_model(MODEL_FILE)

    # Start from a random real sequence, then keep predicting the next note
    pattern = list(X_int[np.random.randint(0, len(X_int))])
    predicted = []
    for _ in range(NUM_NOTES_TO_GENERATE):
        inp = np.reshape(pattern, (1, len(pattern), 1)) / float(vocab_size)
        probs = model.predict(inp, verbose=0)[0]
        idx = sample_with_temperature(probs, TEMPERATURE)
        predicted.append(int_to_note[idx])
        pattern.append(idx)
        pattern = pattern[1:]

    create_midi(predicted)


def create_midi(predicted):
    """Convert predicted note strings back into a MIDI file."""
    offset = 0
    output = []
    for p in predicted:
        if "." in p or p.isdigit():  # chord
            chord_notes = []
            for n in p.split("."):
                new_note = note.Note(int(n))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output.append(new_chord)
        else:  # single note
            new_note = note.Note(p)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output.append(new_note)
        offset += 0.5

    stream.Stream(output).write("midi", fp=OUTPUT_FILE)
    print("Saved generated music to", OUTPUT_FILE)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("train", "generate"):
        sys.exit("Usage: python music_generation.py [train|generate]")
    train() if sys.argv[1] == "train" else generate()
