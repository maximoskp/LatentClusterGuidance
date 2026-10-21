import os
import math
import re
import numpy as np
import pretty_midi
import pandas as pd
from music21 import converter, meter 
from bisect import bisect_right
from tqdm import tqdm

# ---------- HELPER: bar & half-bar segmentation (in seconds) ----------
def _get_bar_segments(pm: pretty_midi.PrettyMIDI):
    """
    Return a list of (bar_start_time, bar_end_time) in seconds.
    Prefers PrettyMIDI downbeats; falls back to 4/4 (every 4 beats).
    """
    end_time = pm.get_end_time()
    bar_starts = []

    downbeats = pm.get_downbeats()
    if len(downbeats) >= 1:
        bar_starts = list(downbeats)
        if bar_starts[0] > 0.0:
            bar_starts = [0.0] + bar_starts
    else:
        beats = pm.get_beats()
        if len(beats) >= 1:
            bar_starts = beats[::4]  # assume 4/4
            if not bar_starts or bar_starts[0] > 0.0:
                bar_starts = [0.0] + bar_starts
        else:
            bar_starts = [0.0]

    bars = []
    for i, s in enumerate(bar_starts):
        e = bar_starts[i+1] if i+1 < len(bar_starts) else end_time
        if e > s:
            bars.append((s, e))
    return bars

def _build_halfbar_table(pm: pretty_midi.PrettyMIDI, chords_with_times):
    """
    Build half-bar segments with an assigned chord:
      returns list of (half_start, half_end, chord_or_None)
    The chord for a half-bar is the chord active at half_start:
    last chord whose onset <= half_start; if none, chord is None.
    """
    bars = _get_bar_segments(pm)
    halfbars = []
    chord_times = [t for t, _ in chords_with_times]
    chord_vals  = [c for _, c in chords_with_times]

    for (s, e) in bars:
        mid = 0.5 * (s + e)
        for hs, he in ((s, mid), (mid, e)):
            idx = bisect_right(chord_times, hs) - 1
            chord = chord_vals[idx] if idx >= 0 else None
            halfbars.append((hs, he, chord))
    return halfbars

def _chord_from_halfbar(halfbar_table, t):
    """
    Return the chord assigned to the half-bar that contains time t.
    If no half-bar contains t (rare at the very end), return None.
    """
    starts = [hs for (hs, _, _) in halfbar_table]
    idx = bisect_right(starts, t) - 1
    if idx >= 0:
        hs, he, chord = halfbar_table[idx]
        if hs <= t < he:
            return chord
    return None

# ---------- HELPER: musical 1/16 windows (tempo-aware fallback) ----------
def _sixteenth_windows(pm: pretty_midi.PrettyMIDI):
    """
    Build musical sixteenth-note windows in seconds.
    Prefer beat-subdivision (each beat -> 4 sixteenths). If beats insufficient,
    fall back to uniform (60/BPM)/4 seconds using median tempo (or 120 BPM).
    """
    windows = []
    beats = pm.get_beats()
    end_time = pm.get_end_time()

    if len(beats) >= 2:
        # Subdivide all complete beat intervals
        for i in range(len(beats) - 1):
            b0, b1 = beats[i], beats[i+1]
            step = (b1 - b0) / 4.0
            for j in range(4):
                windows.append((b0 + j * step, b0 + (j + 1) * step))
        # Tail: from last beat to end_time
        last = beats[-1]
        if end_time > last:
            # approximate last step by previous beat length (or leave as one window)
            prev_len = beats[-1] - beats[-2] if len(beats) >= 2 else (end_time - last)
            step = prev_len / 4.0 if prev_len > 0 else (end_time - last)
            t = last
            while t < end_time - 1e-9:
                windows.append((t, min(t + step, end_time)))
                t += step
    else:
        tempi_times, tempi_vals = pm.get_tempo_changes()
        bpm = float(np.median(tempi_vals)) if len(tempi_vals) > 0 else 120.0
        sixteenth = (60.0 / bpm) / 4.0
        t = 0.0
        while t < end_time - 1e-9:
            windows.append((t, min(t + sixteenth, end_time)))
            t += sixteenth
    return windows


#####################################
# 1. Chord Extraction Functions
#####################################

def extract_chords_from_midi(midi_file_path, track_index=1, grouping_threshold=0.05):
    """
    Extract chords from a MIDI file from the specified track.
    Group together notes that start within 'grouping_threshold' seconds.
    
    Parameters:
        midi_file_path (str): Path to the MIDI file.
        track_index (int): Track index for chords (e.g., 0 = melody, 1 = chords).
        grouping_threshold (float): Time threshold (seconds) to group notes.
    
    Returns:
        List[Tuple[int]]: List of chords (each chord is a tuple of sorted pitch classes).
    """
    pm = pretty_midi.PrettyMIDI(midi_file_path)
    if track_index >= len(pm.instruments):
        raise ValueError(f"Track index {track_index} is out of range for file {midi_file_path}.")
    
    chord_instrument = pm.instruments[track_index]
    sorted_notes = sorted(chord_instrument.notes, key=lambda note: note.start)
    
    chords = []
    current_chord_notes = []
    current_time = None

    for note in sorted_notes:
        if current_time is None:
            current_time = note.start
            current_chord_notes.append(note)
        else:
            if note.start - current_time <= grouping_threshold:
                current_chord_notes.append(note)
            else:
                chord_pitch_classes = tuple(sorted({n.pitch % 12 for n in current_chord_notes}))
                chords.append(chord_pitch_classes)
                current_chord_notes = [note]
                current_time = note.start

    if current_chord_notes:
        chord_pitch_classes = tuple(sorted({n.pitch % 12 for n in current_chord_notes}))
        chords.append(chord_pitch_classes)
    
    return chords

def extract_chords_with_times_from_midi(midi_file_path, track_index=1, grouping_threshold=0.05):
    """
    Extract chords from the specified track of a MIDI file along with their onset times.
    
    Parameters:
        midi_file_path (str): Path to the MIDI file.
        track_index (int): Track index for chords.
        grouping_threshold (float): Time threshold to group notes.
    
    Returns:
        List[Tuple[float, Tuple[int]]]: A list of (onset_time, chord) pairs.
    """
    pm = pretty_midi.PrettyMIDI(midi_file_path)
    if track_index >= len(pm.instruments):
        raise ValueError(f"Track index {track_index} is out of range for file {midi_file_path}.")
    
    chord_instrument = pm.instruments[track_index]
    sorted_notes = sorted(chord_instrument.notes, key=lambda note: note.start)
    
    chords_with_times = []
    current_chord_notes = []
    current_time = None

    for note in sorted_notes:
        if current_time is None:
            current_time = note.start
            current_chord_notes.append(note)
        else:
            if note.start - current_time <= grouping_threshold:
                current_chord_notes.append(note)
            else:
                chord_pitch_classes = tuple(sorted({n.pitch % 12 for n in current_chord_notes}))
                chords_with_times.append((current_time, chord_pitch_classes))
                current_chord_notes = [note]
                current_time = note.start

    if current_chord_notes:
        chord_pitch_classes = tuple(sorted({n.pitch % 12 for n in current_chord_notes}))
        chords_with_times.append((current_time, chord_pitch_classes))
    
    return chords_with_times

#####################################
# 2. Tonal Centroid via Lookup (Harte et al., 2006)
#####################################

def tonal_centroid(notes):
    """
    Compute the 6-D tonal centroid for a set of pitch classes using lookup tables.
    The centroid is computed by averaging three 2-D vectors:
      - Fifths (e.g. a perfect fifth is assigned [1.0, 0.0])
      - Minor thirds
      - Major thirds
    The three 2-D vectors are concatenated to form a 6-D vector.
    
    Parameters:
        notes (list of int): List of pitch class numbers (0-11).
    
    Returns:
        np.ndarray: A 6-D numpy array representing the tonal centroid.
    """
    # Lookup dictionaries:
    fifths_lookup = {
        9: [1.0, 0.0],
        2: [math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)],
        7: [math.cos(2.0 * math.pi / 6.0), math.sin(2.0 * math.pi / 6.0)],
        0: [0.0, 1.0],
        5: [math.cos(4.0 * math.pi / 6.0), math.sin(4.0 * math.pi / 6.0)],
        10: [math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)],
        3: [-1.0, 0.0],
        8: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
        1: [math.cos(8.0 * math.pi / 6.0), math.sin(8.0 * math.pi / 6.0)],
        6: [0.0, -1.0],
        11: [math.cos(10.0 * math.pi / 6.0), math.sin(10.0 * math.pi / 6.0)],
        4: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]
    }
    minor_thirds_lookup = {
        3: [1.0, 0.0],
        7: [1.0, 0.0],
        11: [1.0, 0.0],
        0: [0.0, 1.0],
        4: [0.0, 1.0],
        8: [0.0, 1.0],
        1: [-1.0, 0.0],
        5: [-1.0, 0.0],
        9: [-1.0, 0.0],
        2: [0.0, -1.0],
        6: [0.0, -1.0],
        10: [0.0, -1.0]
    }
    major_thirds_lookup = {
        0: [0.0, 1.0],
        3: [0.0, 1.0],
        6: [0.0, 1.0],
        9: [0.0, 1.0],
        2: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
        5: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
        8: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
        11: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
        1: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
        4: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
        7: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
        10: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]
    }
    
    # Weights for each component:
    r1 = 1.0  # for fifths
    r2 = 1.0  # for minor thirds
    r3 = 0.5  # for major thirds
    
    fifths = [0.0, 0.0]
    minor = [0.0, 0.0]
    major = [0.0, 0.0]
    
    if notes:
        for note in notes:
            for i in range(2):
                fifths[i] += r1 * fifths_lookup[note][i]
                minor[i] += r2 * minor_thirds_lookup[note][i]
                major[i] += r3 * major_thirds_lookup[note][i]
        n = len(notes)
        for i in range(2):
            fifths[i] /= n
            minor[i] /= n
            major[i] /= n
    return np.array(fifths + minor + major)

#####################################
# 3. Metrics
#####################################

def compute_chord_histogram_entropy(chords):
    """
    Compute the Chord Histogram Entropy (CHE) of a chord sequence.
    
    Parameters:
        chords (List[Tuple[int]]): List of chords (each chord is a tuple of pitch classes).
    
    Returns:
        float: Entropy in bits.
    """
    if not chords:
        return 0.0
    histogram = {}
    for chord in chords:
        histogram[chord] = histogram.get(chord, 0) + 1
    total = len(chords)
    entropy = 0.0
    for count in histogram.values():
        p = count / total
        entropy -= p * np.log(p + 1e-6)
    return entropy

def compute_chord_coverage(chords):
    """
    Compute the Chord Coverage (CC) of a chord sequence.
    
    Parameters:
        chords (List[Tuple[int]]): List of chords.
    
    Returns:
        int: Number of unique chords.
    """
    return len(set(chords))

def chord_to_tonal_vector(chord):
    """
    Compute the 6-D tonal vector (tonal centroid) for a chord.
    
    Parameters:
        chord (Tuple[int]): A chord represented as a tuple of pitch classes.
    
    Returns:
        np.ndarray: 6-D tonal vector.
    """
    return tonal_centroid(list(chord))

def chord_tonal_distance(chord1, chord2):
    """
    Compute Euclidean distance between the tonal vectors of two chords.
    
    Parameters:
        chord1, chord2 (Tuple[int]): Chords as tuples of pitch classes.
    
    Returns:
        float: Euclidean distance.
    """
    tv1 = chord_to_tonal_vector(chord1)
    tv2 = chord_to_tonal_vector(chord2)
    return np.linalg.norm(tv1 - tv2)

def compute_chord_tonal_distance(chords):
    """
    Compute average Chord Tonal Distance (CTD) over a sequence of chords.
    
    Parameters:
        chords (List[Tuple[int]]): List of chords.
    
    Returns:
        float: Average CTD.
    """
    if len(chords) < 2:
        return 0.0
    distances = []
    for i in range(1, len(chords)):
        distances.append(chord_tonal_distance(chords[i-1], chords[i]))
    return np.mean(distances)

def compute_ctnctr(midi_file_path, chord_track_index=1, melody_track_index=0, grouping_threshold=0.05):
    """
    CTnCTR aligned with paper:
      - Active chord is the chord label of the *half-bar* containing the melody note.
      - Proper NCT (n_p): an NCT whose immediately following different note is
        a chord tone of the *same* half-bar chord and within ≤ 2 semitones.
      - Score: (n_c + n_p) / (n_c + n_n)
    """
    pm = pretty_midi.PrettyMIDI(midi_file_path)
    if melody_track_index >= len(pm.instruments):
        raise ValueError(f"Melody track index {melody_track_index} out of range for file {midi_file_path}.")

    melody_notes = sorted(pm.instruments[melody_track_index].notes, key=lambda note: note.start)
    chords_with_times = extract_chords_with_times_from_midi(
        midi_file_path, track_index=chord_track_index, grouping_threshold=grouping_threshold
    )
    if not chords_with_times:
        return 0.0

    halfbars = _build_halfbar_table(pm, chords_with_times)

    nc = 0  # chord tones
    nn = 0  # non-chord tones
    np_ = 0 # proper NCTs

    for idx, note in enumerate(melody_notes):
        note_pc = note.pitch % 12
        chord = _chord_from_halfbar(halfbars, note.start)
        if not chord:  # N.C. for this half-bar
            nn += 1
            continue

        if note_pc in chord:
            nc += 1
        else:
            nn += 1
            # proper NCT test against the *same half-bar chord*
            for next_note in melody_notes[idx+1:]:
                if next_note.pitch != note.pitch:
                    if (next_note.pitch % 12) in chord and abs(next_note.pitch - note.pitch) <= 2:
                        np_ += 1
                    break

    denom = nc + nn
    return (nc + np_) / denom if denom > 0 else 0.0


def compute_pcs(midi_file_path, chord_track_index=1, melody_track_index=0, grouping_threshold=0.05):
    """
    PCS aligned with paper:
      - For each melody note, compare to the chord of the *half-bar* containing the note.
      - Force chord tones to the octave at-or-below the melody pitch.
      - Score per interval: +1 for {0,3,4,7,8,9}, 0 for 5, -1 otherwise.
      - Average over 1/16 windows (musical), excluding windows with no notes.
    """
    pm = pretty_midi.PrettyMIDI(midi_file_path)
    chords_with_times = extract_chords_with_times_from_midi(
        midi_file_path, track_index=chord_track_index, grouping_threshold=grouping_threshold
    )
    if not chords_with_times:
        return 0.0

    if melody_track_index >= len(pm.instruments):
        raise ValueError(f"Melody track index {melody_track_index} out of range for file {midi_file_path}.")
    melody_notes = sorted(pm.instruments[melody_track_index].notes, key=lambda note: note.start)

    halfbars = _build_halfbar_table(pm, chords_with_times)
    windows = _sixteenth_windows(pm)

    def consonance_score(interval):
        if interval in {0, 3, 4, 7, 8, 9}:
            return 1
        elif interval == 5:
            return 0
        return -1

    window_scores = []
    for (w_start, w_end) in windows:
        notes_in_window = [n for n in melody_notes if w_start <= n.start < w_end]
        if not notes_in_window:
            continue
        note_scores = []
        for note in notes_in_window:
            chord = _chord_from_halfbar(halfbars, note.start)
            if not chord:
                # N.C. → treat as non-chord: consonance undefined; count as -1 across the board?
                # Paper excludes rests; for N.C. we'll just skip scoring this note.
                # (Alternative would be to score as -1; we choose skip to avoid bias.)
                continue
            chord_tone_scores = []
            for chord_pc in chord:
                k = (note.pitch - chord_pc) // 12
                chord_pitch = chord_pc + 12 * k
                if chord_pitch > note.pitch:
                    chord_pitch -= 12
                interval = note.pitch - chord_pitch
                chord_tone_scores.append(consonance_score(interval))
            if chord_tone_scores:
                note_scores.append(float(np.mean(chord_tone_scores)))
        if note_scores:
            window_scores.append(float(np.mean(note_scores)))

    return float(np.mean(window_scores)) if window_scores else 0.0


def compute_mctd(midi_file_path, chord_track_index=1, melody_track_index=0, grouping_threshold=0.05):
    """
    MCTD aligned with paper:
      - Melody note: singleton PCP → 6-D tonal centroid (Harte).
      - Chord: half-bar chord label → PCs → 6-D centroid.
      - Distance: Euclidean; duration-weighted average over notes.
      - If the half-bar chord is N.C., skip that note (undefined reference).
    """
    pm = pretty_midi.PrettyMIDI(midi_file_path)
    if melody_track_index >= len(pm.instruments):
        raise ValueError(f"Melody track index {melody_track_index} out of range for file {midi_file_path}.")
    melody_notes = sorted(pm.instruments[melody_track_index].notes, key=lambda note: note.start)

    chords_with_times = extract_chords_with_times_from_midi(
        midi_file_path, track_index=chord_track_index, grouping_threshold=grouping_threshold
    )
    if not chords_with_times:
        return 0.0

    halfbars = _build_halfbar_table(pm, chords_with_times)

    weighted_sum = 0.0
    total_dur = 0.0

    for note in melody_notes:
        chord = _chord_from_halfbar(halfbars, note.start)
        if not chord:
            continue  # N.C. half-bar → undefined distance; skip
        melody_tc = tonal_centroid([note.pitch % 12])
        chord_tc  = tonal_centroid(list(chord))
        d = float(np.linalg.norm(melody_tc - chord_tc))
        dur = note.end - note.start
        weighted_sum += d * dur
        total_dur += dur

    return (weighted_sum / total_dur) if total_dur > 0 else 0.0

def compute_HRHE_and_HRC_intervals(chords_with_times, quantization=0.05):
    """
    Compute the Harmonic Rhythm Histogram Entropy (HRHE) and Harmonic Rhythm Coverage (HRC)
    from a list of chord events with onset times.
    
    Parameters:
        chords_with_times (List[Tuple[float, chord]]): List of tuples where each tuple contains
            the onset time (in seconds) and a chord (represented as pitch classes).
        quantization (float): Quantization step (in seconds) used to discretize the inter-chord intervals.
    
    Returns:
        tuple: (HRHE, HRC)
            HRHE (float): The entropy (in nats) of the histogram of quantized inter-chord intervals.
            HRC (int): The number of unique quantized interval types.
    """
    if len(chords_with_times) < 2:
        return 0.0, 0

    # Compute inter-onset intervals between successive chord events.
    intervals = []
    for i in range(1, len(chords_with_times)):
        diff = chords_with_times[i][0] - chords_with_times[i-1][0]
        # Quantize the interval (e.g., rounding to the nearest multiple of 'quantization')
        quantized_diff = round(diff / quantization) * quantization
        intervals.append(quantized_diff)
    # print(intervals)
    
    # Build a histogram (frequency count) of the quantized intervals.
    hist = {}
    for diff in intervals:
        hist[diff] = hist.get(diff, 0) + 1

    total = sum(hist.values())
    HRHE = 0.0
    for count in hist.values():
        p = count / total
        HRHE -= p * np.log(p + 1e-6)  # using a small epsilon to avoid log(0)
    
    # HRC: number of unique quantized interval types.
    HRC = len(hist)
    return HRHE, HRC

def compute_HRHE_and_HRC_onsets(midi_file_path, chord_part_index=1, quantization=0.05):
    """
    Compute the Harmonic Rhythm Histogram Entropy (HRHE) and Harmonic Rhythm Coverage (HRC)
    based on chord offsets (relative to the measure start) using music21.
    
    This function:
      1. Parses the MIDI file using music21 and selects the chord part.
      2. Extracts measures from the chord part.
      3. For each measure, extracts chord events and their offsets (in quarter-note units) within the measure.
      4. Quantizes these offsets using the given quantization step.
      5. Builds a histogram of the quantized offsets.
      6. Computes the entropy (HRHE, in nats) of this histogram and counts the number of unique offset types (HRC).
    
    Parameters:
        midi_file_path (str): Path to the MIDI file.
        chord_part_index (int): Index of the part containing chords.
        quantization (float): Quantization step (in quarter-note units) for the offsets.
    
    Returns:
        tuple: (HRHE, HRC)
            HRHE (float): The entropy (in nats) of the histogram of quantized chord offsets.
            HRC (int): The number of unique quantized offset types.
    """
    # Parse the score and select the chord part.
    score = converter.parse(midi_file_path)
    try:
        chord_part = score.parts[chord_part_index]
    except IndexError:
        raise ValueError(f"Chord part index {chord_part_index} is out of range.")
    
    # Extract measures from the chord part.
    measures = chord_part.getElementsByClass('Measure')
    if not measures:
        raise ValueError("No measures found in the chord part.")
    
    # Extract chord offsets (relative to measure start, in quarter-note units)
    offsets = []
    for m in measures:
        chords_in_measure = m.getElementsByClass('Chord')
        for ch in chords_in_measure:
            offsets.append(ch.offset)
    
    if not offsets:
        return 0.0, 0
    
    # Quantize the offsets.
    quantized_offsets = [round(o / quantization) * quantization for o in offsets]
    
    # Build a histogram (frequency count) of the quantized offsets.
    hist = {}
    for q in quantized_offsets:
        hist[q] = hist.get(q, 0) + 1
    
    # Compute the entropy (HRHE).
    total = sum(hist.values())
    HRHE = 0.0
    for count in hist.values():
        p = count / total
        HRHE -= p * np.log(p + 1e-6)  # small epsilon to avoid log(0)
    
    # HRC is simply the number of unique quantized offset types.
    HRC = len(hist)
    
    return HRHE, HRC

def compute_CBS(midi_file_path, chord_part_index=1, tolerance=0.05):
    """
    Compute the Chord Beat Strength (CBS) for a MIDI file.
    
    This implementation uses music21 to extract measures and detect the time signature 
    for each measure. For each chord event (extracted by chordifying each measure), the 
    onset is measured relative to the start of the measure (in quarter-note units).
    
    The ideal positions are defined on a 16th-note grid:
      - For a measure with time signature A/B:
          • Beat duration (in quarter notes) is: beat_duration = 4 / B.
          • The number of beats in the measure is A.
      - Strong beats are at offsets: 0, 1×beat_duration, 2×beat_duration, …, (A-1)×beat_duration.
      - Half-beats are at: (i + 0.5)×beat_duration for i = 0,1,...,A-1.
      - Eighth-note subdivisions are at: (i + 0.25)×beat_duration and (i + 0.75)×beat_duration.
    
    Scoring is defined as follows (using a 4/4 example):
      - Score 0 if the chord is at the beginning of the measure (offset 0),
      - Score 1 if the chord is on any other strong beat (e.g., 1.0, 2.0, or 3.0),
      - Score 2 if the chord is on a half-beat (e.g., 0.5, 1.5, etc.),
      - Score 3 if the chord is on an eighth-note subdivision (e.g., 0.25, 0.75, etc.),
      - Score 4 for all other positions.
    
    The final CBS is the average of these scores over all chord events.
    
    Parameters:
        midi_file_path (str): Path to the MIDI file.
        chord_part_index (int): Index of the part containing chords.
        tolerance (float): Tolerance (in quarter notes) to consider an event "on" an ideal position.
    
    Returns:
        float: The average CBS score.
    """
    # Parse the MIDI file using music21.
    score = converter.parse(midi_file_path)
    
    # Select the chord part.
    try:
        chord_part = score.parts[chord_part_index]
    except IndexError:
        raise ValueError(f"Chord part index {chord_part_index} is out of range.")
    
    # Extract measures from the chord part.
    measures = chord_part.getElementsByClass('Measure')
    if not measures:
        raise ValueError("No measures found in the chord part.")
    
    def ideal_positions_for_measure(ts):
        """
        Given a TimeSignature object, return three lists:
          - strong: positions (in quarter notes) of strong beats,
          - half: positions for half-beats,
          - eighth: positions for eighth-note subdivisions.
        """
        num_beats = ts.numerator
        beat_duration = 4 / ts.denominator  # Duration of one beat in quarter notes.
        measure_duration = num_beats * beat_duration
        
        strong = [i * beat_duration for i in range(num_beats)]
        half = [i * beat_duration + 0.5 * beat_duration for i in range(num_beats)]
        
        eighth = []
        for i in range(num_beats):
            pos1 = i * beat_duration + 0.25 * beat_duration
            pos2 = i * beat_duration + 0.75 * beat_duration
            if pos1 < measure_duration:
                eighth.append(pos1)
            if pos2 < measure_duration:
                eighth.append(pos2)
        return strong, half, eighth
    
    scores = []
    for m in measures:
        # Get the time signature for the measure. Use the first if there are multiple.
        ts_list = m.getTimeSignatures()
        ts = ts_list[0] if ts_list else meter.TimeSignature('4/4')
        
        strong, half, eighth = ideal_positions_for_measure(ts)
        
        # Chordify the measure to collapse simultaneous events into chord events.
        m_chords = m.chordify().recurse().getElementsByClass('Chord')
        for ch in m_chords:
            offset = ch.offset  # offset within the measure (in quarter notes)
            # New scoring: 
            # 0 if exactly at beginning of measure, 1 if on other strong beats,
            # 2 if on half-beat, 3 if on eighth subdivision, 4 otherwise.
            if abs(offset - strong[0]) <= tolerance:
                score_val = 0
            elif any(abs(offset - pos) <= tolerance for pos in strong[1:]):
                score_val = 1
            elif any(abs(offset - pos) <= tolerance for pos in half):
                score_val = 2
            elif any(abs(offset - pos) <= tolerance for pos in eighth):
                score_val = 3
            else:
                score_val = 4
            scores.append(score_val)
    
    return np.mean(scores) if scores else 0.0

#####################################
# Pipeline: Compute All Metrics for a Single MIDI File
#####################################

def compute_all_metrics(midi_file_path, chord_track_index=1, melody_track_index=0, grouping_threshold=0.05):
    """
    Compute all metrics for a given MIDI file.
    
    Metrics:
      - CHE: Chord Histogram Entropy
      - CC: Chord Coverage
      - CTD: Chord Tonal Distance (lookup-based)
      - HRHE_i: Harmonic Rhythm Histogram Entropy (inter-chord intervals)
      - HRC_i: Harmonic Rhythm Coverage
      - CBS: Chord Beat Strength
      - CTnCTR: Chord Tone to Non-Chord Tone Ratio
      - PCS: Pitch Consonance Score
      - MCTD: Melody-chord Tonal Distance
    
    Parameters:
        midi_file_path (str): Path to the MIDI file.
        chord_track_index (int): Track index for chords.
        melody_track_index (int): Track index for melody.
        grouping_threshold (float): Time threshold for chord grouping.
    
    Returns:
        dict: Dictionary with keys for each metric.
    """
    chords = extract_chords_from_midi(midi_file_path, track_index=chord_track_index, grouping_threshold=grouping_threshold)
    chords_with_times = extract_chords_with_times_from_midi(midi_file_path, track_index=chord_track_index, grouping_threshold=grouping_threshold)


    metrics = {}
    metrics["CHE"] = compute_chord_histogram_entropy(chords)
    metrics["CC"] = compute_chord_coverage(chords)
    metrics["CTD"] = compute_chord_tonal_distance(chords)
    metrics["HRHE_i"], metrics["HRC_i"] = compute_HRHE_and_HRC_intervals(chords_with_times, grouping_threshold)
    metrics["CBS"] = compute_CBS(midi_file_path, chord_track_index, grouping_threshold)
    metrics["CTnCTR"] = compute_ctnctr(midi_file_path, chord_track_index, melody_track_index, grouping_threshold)
    metrics["PCS"] = compute_pcs(midi_file_path, chord_track_index, melody_track_index, grouping_threshold)
    metrics["MCTD"] = compute_mctd(midi_file_path, chord_track_index, melody_track_index, grouping_threshold)
    
    return metrics



def _file_id_from_name(filename):
    """
    Normalize a MIDI filename into a pairing ID.
    Handles both prefix style (real_0.mid / gen_0.mid) and suffix style
    (foo_real.mid / foo_gen.mid). If trailing digits exist, uses them.
    """
    name = os.path.splitext(os.path.basename(filename))[0].lower()

    # strip common prefixes
    if name.startswith(("real_", "gen_", "gt_", "pred_", "fake_", "ref_", "target_")):
        for p in ("real_", "gen_", "gt_", "pred_", "fake_", "ref_", "target_"):
            if name.startswith(p):
                name = name[len(p):]
                break

    # strip common suffixes
    for s in ("_real", "_gen", "_gt", "_pred", "_fake"):
        if name.endswith(s):
            name = name[: -len(s)]
            break

    # if there are trailing digits, use them as the ID (robust across styles)
    m = re.search(r'(\d+)$', name)
    return m.group(1) if m else name

def _index_midis(folder):
    """
    Return dict: id -> fullpath for all .mid files under 'folder' (non-recursive).
    """
    out = {}
    for f in os.listdir(folder):
        if f.lower().endswith('.mid'):
            fid = _file_id_from_name(f)
            out[fid] = os.path.join(folder, f)
    return out

def _pair_real_and_gen(ground_truth_folder, gen_folder):
    """
    Build pairs between real files (in ground_truth_folder) and generated files (in gen_folder).
    Matching is by the normalized ID (_file_id_from_name).
    Returns dict: id -> (real_path, gen_path)
    """
    real_map = _index_midis(ground_truth_folder)
    gen_map  = _index_midis(gen_folder)
    common = sorted(set(real_map.keys()) & set(gen_map.keys()))
    if not common:
        print(f"No matching IDs between real='{ground_truth_folder}' and gen='{gen_folder}'.")
    pairs = {fid: (real_map[fid], gen_map[fid]) for fid in common}
    # Warn about missing items to help debug
    missing_in_gen = sorted(set(real_map.keys()) - set(gen_map.keys()))
    missing_in_real = sorted(set(gen_map.keys()) - set(real_map.keys()))
    if missing_in_gen:
        print(f"  [warn] {len(missing_in_gen)} real files have no generated match in '{gen_folder}': {missing_in_gen[:5]}{' ...' if len(missing_in_gen) > 5 else ''}")
    if missing_in_real:
        print(f"  [warn] {len(missing_in_real)} generated files have no real match in '{ground_truth_folder}': {missing_in_real[:5]}{' ...' if len(missing_in_real) > 5 else ''}")
    return pairs

# ---------- REPLACEMENT FOR compute_metrics_for_pairs ----------
def compute_metrics_for_pairs_separated(ground_truth_folder, gen_folder, chord_track_index=1, melody_track_index=0, grouping_threshold=0.05):
    """
    Compute aggregated metrics for *paired* real-vs-generated files, where
    real files live in 'ground_truth_folder' and generated files live in 'gen_folder'.

    Returns:
        dict with:
          - 'real': aggregated stats (mean/std/count) over matched real files
          - 'gen' : aggregated stats (mean/std/count) over matched generated files
          - '__pairs': list of per-pair dicts:
                [{'id': <pair_id>, 'real': {metric->value}, 'gen': {metric->value}}, ...]
    """
    pairs = _pair_real_and_gen(ground_truth_folder, gen_folder)
    metrics_by_group = {"real": [], "gen": []}
    pair_records = []

    for fid, (real_file, gen_file) in tqdm(pairs.items()):
        try:
            m_real = compute_all_metrics(real_file, chord_track_index, melody_track_index, grouping_threshold)
            m_gen  = compute_all_metrics(gen_file,  chord_track_index, melody_track_index, grouping_threshold)
        except Exception as e:
            print(f"Error processing ID {fid}: {e}")
            continue
        metrics_by_group["real"].append(m_real)
        metrics_by_group["gen"].append(m_gen)
        pair_records.append({"id": fid, "real": m_real, "gen": m_gen})

    results = {}
    for group in ["gen", "real"]:
        if metrics_by_group[group]:
            metric_names = metrics_by_group[group][0].keys()
            group_results = {}
            for metric in metric_names:
                values = [m[metric] for m in metrics_by_group[group]]
                group_results[metric] = {
                    "mean": float(np.mean(values)),
                    "std":  float(np.std(values)),
                    "count": len(values)
                }
            results[group] = group_results
        else:
            results[group] = {}
    results["__pairs"] = pair_records

    # Optional: print summary to console
    for group in ["gen", "real"]:
        label = os.path.basename(gen_folder) if group == "gen" else "GROUND TRUTH"
        print(f"{group.upper()} files [{label}]:")
        if results[group]:
            for metric, stats in results[group].items():
                print(f"  {metric}: Mean = {stats['mean']:.3f}, Std = {stats['std']:.3f} (n = {stats['count']})")
        else:
            print("  No files found.")
    return results


def mae_row_from_pairs(instance_name, pair_records):
    """
    Build a row: {"Instance": instance_name, <metric>: mean(|gen - real|)} using
    per-pair metrics already computed. No recomputation of metrics.
    """
    row = {"Instance": instance_name}
    if not pair_records:
        return row
    metric_names = list(pair_records[0]["real"].keys())
    for metric in metric_names:
        diffs = []
        for rec in pair_records:
            vr = rec["real"].get(metric, np.nan)
            vg = rec["gen"].get(metric,  np.nan)
            if np.isfinite(vr) and np.isfinite(vg):
                diffs.append(abs(vg - vr))
        row[metric] = float(np.mean(diffs)) if diffs else np.nan
    return row


def highlight_lowest(series):
    metric_columns = ["CHE", "CC", "CTD", "CTnCTR", "PCS", "MCTD", "HRHE_i", "HRC_i", "CBS"]
    if series.name not in metric_columns or series.dropna().empty:
        return ['' for _ in series]
    min_val = series.min(skipna=True)
    return ['background-color: lightgreen' if (not pd.isna(v) and v == min_val) else '' for v in series]



def highlight_closest(series):
    """
    For a metric column (e.g. "CHE_mean"), use the first row as the ground truth
    value and mark (with a lightgreen background) the cell(s) in the generated rows
    that are closest (in absolute difference) to that ground truth.
    """
    metric_columns = ["CHE", "CC", "CTD", "CTnCTR", "PCS", "MCTD", "HRHE_i", "HRC_i", "CBS"]
    if series.name not in metric_columns:
        return ['' for _ in series]
    ground_truth = series.iloc[0]
    # Compute differences for generated rows only (rows with index >= 1)
    diffs = series.iloc[1:].apply(lambda x: abs(x - ground_truth))
    if diffs.empty:
        return ['' for _ in series]
    min_diff = diffs.min()
    styles = []
    for i, val in enumerate(series):
        if i == 0:
            styles.append('')  # Do not highlight the ground truth row
        else:
            if abs(val - ground_truth) == min_diff:
                styles.append('background-color: lightgreen')
            else:
                styles.append('')
    return styles

# ---------- REPLACEMENT FOR compute_model_results ----------
def compute_setup_results(
        setup_path,
        source_folder,
        target_folder,
        chord_track_index=1,
        melody_track_index=0,
        grouping_threshold=0.05
    ):
    """
    Return:
      - rows_means: first row Ground Truth mean (all real files), then instance means
      - rows_mae:   one row per instance with mean absolute error vs matched ground truth
    """
    rows_means, rows_mae = [], []
    # SOURCE
    source_map = _index_midis(source_folder)
    if not source_map:
        print(f"No source MIDI files found in '{source_folder}'.")
        return rows_means, rows_mae
    # Aggregate ground truth metrics
    source_metrics = []
    for _, source_fp in tqdm(sorted(source_map.items())):
        try:
            m_source = compute_all_metrics(source_fp, chord_track_index, melody_track_index, grouping_threshold)
            source_metrics.append(m_source)
        except Exception as e:
            print(f"  [GT warn] Failed on {source_fp}: {e}")
    if source_metrics:
        gt_row = {"Instance": "Ground Truth"}
        for metric in source_metrics[0].keys():
            vals = [m[metric] for m in source_metrics]
            gt_row[metric] = float(np.mean(vals)) if vals else np.nan
        rows_means.append(gt_row)
    else:
        print("No valid ground-truth metrics computed.")
    
    # TARGET
    target_map = _index_midis(target_folder)
    if not target_map:
        print(f"No real MIDI files found in '{target_folder}'.")
        return rows_means, rows_mae

    # Aggregate ground truth metrics
    target_metrics = []
    for _, target_fp in tqdm(sorted(target_map.items())):
        try:
            m_target = compute_all_metrics(target_fp, chord_track_index, melody_track_index, grouping_threshold)
            target_metrics.append(m_target)
        except Exception as e:
            print(f"  [Target warn] Failed on {target_fp}: {e}")
    if target_metrics:
        gt_row = {"Instance": "Target"}
        for metric in target_metrics[0].keys():
            vals = [m[metric] for m in target_metrics]
            gt_row[f"{metric}"] = float(np.mean(vals)) if vals else np.nan
        rows_means.append(gt_row)
    else:
        print("No valid target metrics computed.")

    # Instances
    for instance_folder in sorted(os.listdir(setup_path)):
        instance_path = os.path.join(setup_path, instance_folder)
        if not os.path.isdir(instance_path):
            continue
        print(f"  Processing instance: {instance_folder}")

        try:
            results = compute_metrics_for_pairs_separated(
                ground_truth_folder=target_folder,
                gen_folder=instance_path,
                chord_track_index=chord_track_index,
                melody_track_index=melody_track_index,
                grouping_threshold=grouping_threshold
            )
        except Exception as e:
            print(f"    Error in instance '{instance_folder}': {e}")
            continue

        # Mean row (as before)
        if "gen" in results and results["gen"]:
            gen_row = {"Instance": instance_folder}
            for metric, stats in results["gen"].items():
                gen_row[metric] = stats.get("mean", np.nan)
            rows_means.append(gen_row)
        else:
            print(f"    No generated metrics for instance '{instance_folder}'.")

        # MAE row from cached per-pair metrics (no recomputation)
        pair_records = results.get("__pairs", [])
        rows_mae.append(mae_row_from_pairs(instance_folder, pair_records))

    return rows_means, rows_mae


# ---------- REPLACEMENT FOR compute_and_save_html_results_by_model ----------
def compute_and_save_html_results_by_setup(
    root_folder,
    source_folder,
    target_folder,
    chord_track_index=1,
    melody_track_index=0,
    grouping_threshold=0.05,
    output_html="results_by_setup.html"
):
    html_blocks = []

    metric_cols = ["CHE", "CC", "CTD", "CTnCTR", "PCS", "MCTD", "HRHE_i", "HRC_i", "CBS"]
    df_means_full = pd.DataFrame(columns=["Instance"] + metric_cols)
    df_mae_full = pd.DataFrame(columns=["Instance"] + metric_cols)

    for setup_name in sorted(os.listdir(root_folder)):
        setup_path = os.path.join(root_folder, setup_name)
        if not os.path.isdir(setup_path):
            continue

        print(f"Processing setup: {setup_name}")
        rows_means, rows_mae = compute_setup_results(
            setup_path=setup_path,
            source_folder=source_folder,
            target_folder=target_folder,
            chord_track_index=chord_track_index,
            melody_track_index=melody_track_index,
            grouping_threshold=grouping_threshold
        )
        if not rows_means and not rows_mae:
            print(f"  No valid data for setup {setup_name}")
            continue

        df_means = pd.DataFrame(rows_means)
        df_means_full = pd.concat([df_means_full, df_means], ignore_index=True)
        df_mae = pd.DataFrame(rows_mae)
        df_mae_full = pd.concat([df_mae_full, df_mae], ignore_index=True)

        # Table 1: Means (Ground Truth on first row; highlight closest-to-GT)
        means_html = ""
        if rows_means:
            df_means = pd.DataFrame(rows_means)
            present_cols_means = [c for c in metric_cols if c in df_means.columns]
            styled_means = df_means.style.apply(highlight_closest, subset=present_cols_means)
            styled_means = styled_means.set_caption(
                f"<h3>Setup: {setup_name} — Mean Absolute Error</h3>"
            )
            means_html = styled_means.to_html()

        # Table 2: MAE (instances only; highlight lowest)
        mae_html = ""
        if rows_mae:
            df_mae = pd.DataFrame(rows_mae)
            present_cols_mae = [c for c in metric_cols if c in df_mae.columns]
            styled_mae = df_mae.style.apply(highlight_lowest, subset=present_cols_mae)
            styled_mae = styled_mae.set_caption(
                f"<h3>Setup: {setup_name} — Mean Absolute Error</h3>"
            )
            mae_html = styled_mae.to_html()

        html_blocks.append(means_html + mae_html)

    legend_html = """
    <div style="margin-bottom: 30px;">
      <h2>Metrics Information</h2>
      <ul>
         <li><strong>CHE</strong>: Chord Histogram Entropy</li>
         <li><strong>CC</strong>: Chord Coverage</li>
         <li><strong>CTD</strong>: Chord Tonal Distance</li>
         <li><strong>CTnCTR</strong>: Chord Tone to Non-Chord Tone Ratio</li>
         <li><strong>PCS</strong>: Pitch Consonance Score</li>
         <li><strong>MCTD</strong>: Melody-chord Tonal Distance</li>
         <li><strong>HRHE_i</strong>: Harmonic Rhythm Entropy (intervals)</li>
         <li><strong>HRC_i</strong>: Harmonic Rhythm Coverage</li>
         <li><strong>CBS</strong>: Chord Beat Strength</li>
      </ul>
      <p><em>MAE table shows mean absolute difference to ground truth across matched files. Lower is better.</em></p>
    </div>
    """

    full_html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>Metrics Results by Setup/Instance</title>
        <style>
          table {{ border-collapse: collapse; margin-bottom: 30px; }}
          th, td {{ border: 1px solid black; padding: 5px; text-align: center; }}
          caption {{ caption-side: top; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
          h3 {{ margin-top: 0; }}
        </style>
      </head>
      <body>
        {legend_html}
        {"".join(html_blocks)}
      </body>
    </html>
    """
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"HTML results saved to {output_html}")
    df_means_full.to_csv(output_html.replace('.html', '_means.csv'), index=False)
    print(f"Means CSV results saved to {output_html.replace('.html', '_means.csv')}")
    metric_cols = df_mae_full.columns.drop('Instance')
    df_mae_full['avg'] = df_mae_full[metric_cols].mean(axis=1)
    df_mae_full.to_csv(output_html.replace('.html', '_mae.csv'), index=False)
    latex_table = df_mae_full.to_latex(float_format="%.4f")
    with open(output_html.replace('.html', '_mae.tex'), "w") as f:
        f.write(latex_table)
    print(f"MAE CSV results saved to {output_html.replace('.html', '_mae.csv')}")

if __name__ == "__main__":
    source_path = "./MIDIs/jazz2nott/real"          # folder with the source/melody MIDIs
    target_path = "./MIDIs/nott2jazz/real"          # folder with the target/harmony MIDIs
    root_folder = "./MIDIs/jazz2nott/gen"   # folder containing setups; each setup has instance folders

    compute_and_save_html_results_by_setup(
        root_folder=root_folder,
        source_folder=source_path,
        target_folder=target_path,
        chord_track_index=1,
        melody_track_index=0,
        grouping_threshold=0.05,
        output_html="results/results_MAE_jazz2nott.html"
    )

    source_path = "./MIDIs/nott2jazz/real"          # folder with the source/melody MIDIs
    target_path = "./MIDIs/jazz2nott/real"          # folder with the target/harmony MIDIs
    root_folder = "./MIDIs/nott2jazz/gen"   # folder containing setups; each setup has instance folders

    compute_and_save_html_results_by_setup(
        root_folder=root_folder,
        source_folder=source_path,
        target_folder=target_path,
        chord_track_index=1,
        melody_track_index=0,
        grouping_threshold=0.05,
        output_html="results/results_MAE_nott2jazz.html"
    )