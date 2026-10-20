# CodeAlpha_MusicGeneration

Music generation with AI: an LSTM network learns note patterns from MIDI files and composes new music.

## Setup
```
pip install -r requirements.txt
```
Put 20-100 `.mid` files (classical, jazz, etc.) in a folder named `midi_songs/`.
Free datasets: search "Classical Piano MIDI Page" or the Kaggle "Classical Music MIDI" dataset.

## Run
```
python music_generation.py train      # preprocess + train (saves model.keras)
python music_generation.py generate   # creates generated_music.mid
```

## How it works
1. **Preprocess**: music21 reads each MIDI file and extracts notes and chords as text tokens.
2. **Sequences**: every 50 notes become an input; the next note is the target.
3. **Model**: 3 stacked LSTM layers + dropout + softmax over all unique notes.
4. **Generate**: start from a random seed, predict the next note repeatedly (with temperature sampling), then write the notes to a MIDI file.

## Play the output as audio
Open `generated_music.mid` in VLC / MuseScore, or convert it with an online MIDI-to-MP3 tool or FluidSynth.
