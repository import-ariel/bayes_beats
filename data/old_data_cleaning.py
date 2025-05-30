import os
import json
import logging
from music21 import converter, note, chord
from typing import List, Tuple, Optional
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaning:
    """
    DataCleaning class for preprocessing a directory of MIDI files into
    sequences and vocabularies for infinite lo-fi music generation.

    Attributes:
        midi_dir (str): Directory containing MIDI files.
        output_dir (str): Directory for saving JSON outputs.
        symbol_to_int (dict): Mapping of symbol to integer ID.
        duration_to_int (dict): Mapping of duration (as string) to integer ID.
        encoded_sequences (List[List[Tuple[int, int]]]): Encoded data.
    """
    def __init__(self, midi_dir: str, output_dir: str = "processed_lofi_data"):
        self.midi_dir = midi_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.symbol_to_int: dict = {}
        self.duration_to_int: dict = {}
        self.encoded_sequences: List[List[Tuple[int, int]]] = []

    def get_midi_files(self) -> List[str]:
        """Return list of .mid/.midi file paths in midi_dir."""
        midi_files = [
            os.path.join(self.midi_dir, f)
            for f in os.listdir(self.midi_dir)
            if f.lower().endswith(('.mid', '.midi'))
        ]
        logger.info(f"Found {len(midi_files)} MIDI files in {self.midi_dir}.")
        return midi_files

    def parse_midi_files(self, midi_files: List[str]) -> List[List[Tuple[str, float]]]:
        """Parse each MIDI file into a sequence of (symbol, duration)."""
        event_sequences: List[List[Tuple[str, float]]] = []
        for file_path in tqdm(midi_files, desc="Parsing MIDI files"):
            try:
                score = converter.parse(file_path, format="midi")
                sequence: List[Tuple[str, float]] = []
                for element in score.flatten().notesAndRests:
                    if isinstance(element, note.Note):
                        symbol = str(element.pitch)
                    elif isinstance(element, chord.Chord):
                        symbol = ".".join(sorted(str(n) for n in element.normalOrder))
                    elif isinstance(element, note.Rest):
                        symbol = "r"
                    else:
                        continue
                    duration = float(element.quarterLength)
                    sequence.append((symbol, duration))
                if sequence:
                    event_sequences.append(sequence)
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")
        logger.info(f"Parsed {len(event_sequences)} sequences from MIDI files.")
        return event_sequences

    def build_vocabularies(self, event_sequences: List[List[Tuple[str, float]]]) -> None:
        """Build mappings from symbols and durations to integer IDs."""
        all_symbols = sorted({sym for seq in event_sequences for sym, _ in seq})
        all_durations = sorted({dur for seq in event_sequences for _, dur in seq})
        self.symbol_to_int = {s: i for i, s in enumerate(all_symbols)}
        self.duration_to_int = {str(d): i for i, d in enumerate(all_durations)}
        logger.info(f"Built symbol vocab ({len(self.symbol_to_int)}) and duration vocab ({len(self.duration_to_int)}).")

    def encode_sequences(self, event_sequences: List[List[Tuple[str, float]]]) -> None:
        """Encode sequences as lists of (symbol_id, duration_id)."""
        self.encoded_sequences = []
        for seq in event_sequences:
            encoded: List[Tuple[int, int]] = []
            for symbol, duration in seq:
                sym_id = self.symbol_to_int[symbol]
                dur_id = self.duration_to_int[str(duration)]
                encoded.append((sym_id, dur_id))
            self.encoded_sequences.append(encoded)
        logger.info(f"Encoded {len(self.encoded_sequences)} sequences.")

    def save_data(self) -> None:
        """Save encoded sequences and vocabularies as JSON files, with logging."""
        seq_path = os.path.join(self.output_dir, "encoded_sequences.json")
        sym_path = os.path.join(self.output_dir, "symbol_to_int.json")
        dur_path = os.path.join(self.output_dir, "duration_to_int.json")

        with open(seq_path, "w") as f:
            json.dump(self.encoded_sequences, f)
        logger.info(f"Saved encoded sequences to: {seq_path}")

        with open(sym_path, "w") as f:
            json.dump(self.symbol_to_int, f)
        logger.info(f"Saved symbol vocabulary to: {sym_path}")

        with open(dur_path, "w") as f:
            json.dump(self.duration_to_int, f)
        logger.info(f"Saved duration vocabulary to: {dur_path}")

    def run(self, save: bool = True) -> Tuple[List[List[Tuple[int, int]]], dict, dict]:
        """
        Execute the preprocessing pipeline.

        Args:
            save (bool): If True, save data to JSON files in output_dir.

        Returns:
            encoded_sequences, symbol_to_int, duration_to_int
        """
        midi_files = self.get_midi_files()
        event_sequences = self.parse_midi_files(midi_files)
        self.build_vocabularies(event_sequences)
        self.encode_sequences(event_sequences)
        if save:
            self.save_data()
        return self.encoded_sequences, self.symbol_to_int, self.duration_to_int

# Example usage in a notebook or pipeline:
