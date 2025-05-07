#!/usr/bin/env python3
"""
Parallel gender, age, and race inference for images using DeepFace.

Scans a specified input directory for images (png, jpg, jpeg, bmp, tiff),
infers gender, age, and dominant race for detected faces using DeepFace,
and logs results (including status) to a single CSV file. Features include:

- Parallel processing using ThreadPoolExecutor for faster execution.
- Progress logging to console and detailed logging to a file.
- Ability to resume processing by skipping images already present in the output CSV.
- Explicit handling and logging for images where no face is detected or errors occur.
- Configurable number of worker threads and verbosity level.
"""

import os
import sys
import argparse
import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set, List, Tuple, Dict, Any # Kept Dict/Any for analyze_image return type hint clarity

# Attempt to import DeepFace, provide guidance if missing
try:
    from deepface import DeepFace
except ImportError:
    print("Error: DeepFace library not found.")
    print("Please install it: pip install deepface")
    sys.exit(1)

# --- Configuration ---
# Define supported image extensions (case-insensitive)
IMAGE_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Determine project root for potential imports (if running as a script within a larger project)
# This might not be strictly necessary if the script is run standalone or the package is installed.
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    # Insert project root at the beginning of the path
    sys.path.insert(0, _PROJECT_ROOT)


# --- Logger Setup ---
def setup_logger(logfile: str, console_level: int = logging.INFO, file_level: int = logging.DEBUG) -> logging.Logger:
    """Configures logging to both file and console.

    Args:
        logfile (str): Path to the log file.
        console_level (int): Logging level for console output (e.g., logging.INFO).
        file_level (int): Logging level for file output (e.g., logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("infer_attributes")
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(file_level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(threadName)s - %(message)s")
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    return logger


# --- Image Analysis Function ---
# Note: Still returns race_scores dictionary internally, but it won't be used in the CSV
def analyze_image(img_path: str) -> Tuple[str, str, str, str, str, Dict[str, float]]:
    """
    Analyzes a single image using DeepFace for gender, age, and race.

    Uses DeepFace.analyze with enforce_detection=False. Analyzes the first detected face.

    Args:
        img_path (str): The full path to the image file.

    Returns:
        tuple: (img_path, status, gender, age, dominant_race, race_scores)
               - status: "OK", "NoFace", or "Error: <error details>"
               - gender: Dominant gender if found, else "".
               - age: Age if found, else "".
               - dominant_race: Dominant race if found, else "".
               - race_scores: Dict of race scores (returned but not used in CSV).
    """
    logger = logging.getLogger("infer_attributes")
    try:
        # Still need to request 'race' action to get 'dominant_race'
        analysis_result = DeepFace.analyze(
            img_path=img_path,
            actions=["age", "gender", "race"],
            enforce_detection=False,
            silent=True
        )

        if isinstance(analysis_result, list) and len(analysis_result) > 0:
            analysis = analysis_result[0]

            gender = analysis.get("dominant_gender", analysis.get("gender", ""))
            age = analysis.get("age", "")
            dominant_race = analysis.get("dominant_race", "")
            race_scores = analysis.get("race", {}) # Get scores even if unused later

            region = analysis.get("region", {})
            if (region.get("w", 0) == 0 and region.get("h", 0) == 0) or \
               (not gender and not age and not dominant_race): # Adjusted check
                 logger.debug(f"No face detected or no primary attributes found in: {img_path}")
                 return img_path, "NoFace", "", "", "", {}
            else:
                 logger.debug(f"Analysis OK for {img_path}: Gender={gender}, Age={age}, Race={dominant_race}")
                 return img_path, "OK", str(gender), str(age), str(dominant_race), race_scores

        else:
            logger.warning(f"Unexpected result format or no face detected in {img_path}. Result: {analysis_result}")
            return img_path, "NoFace", "", "", "", {}

    except Exception as e:
        logger.error(f"Error analyzing {img_path}: {e}", exc_info=True)
        error_msg = f"Error: {type(e).__name__}"
        return img_path, error_msg, "", "", "", {}


# --- Main Execution Logic ---
def main(input_dir: str, output_csv: str, log_file: str, max_workers: int, verbose: bool) -> None:
    """Main function to orchestrate the image processing pipeline."""

    # Console logging: verbose -> DEBUG, default -> minimal warnings/errors
    console_log_level = logging.DEBUG if verbose else logging.WARNING
    logger = setup_logger(log_file, console_level=console_log_level)

    if not os.path.isdir(input_dir):
        logger.critical(f"Input directory not found or is not a directory: {input_dir}")
        sys.exit(1)

    logger.info(f"Scanning for images in directory: {input_dir}")
    all_images: List[str] = []
    try:
        for root, _, files in os.walk(input_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    full_path = os.path.join(root, f)
                    all_images.append(full_path)
    except Exception as e:
        logger.critical(f"Error scanning directory {input_dir}: {e}")
        sys.exit(1)


    if not all_images:
        logger.warning(f"No images found with extensions {IMAGE_EXTS} in {input_dir}. Exiting.")
        return
    logger.info(f"Found {len(all_images)} potential image files.")

    processed_paths: Set[str] = set()
    if os.path.exists(output_csv):
        try:
            with open(output_csv, "r", newline="", encoding='utf-8') as f_read:
                reader = csv.reader(f_read)
                header = next(reader, None)
                # Check for expected header structure (at least image_path)
                if header and header[0].lower() == "image_path":
                    image_path_index = 0
                    for row in reader:
                        if row and len(row) > image_path_index:
                            processed_paths.add(row[image_path_index])
                    logger.info(f"Found {len(processed_paths)} images already in {output_csv}. Resuming.")
                    print(f"🗃️  Skipping {len(processed_paths)} already processed images")
                elif header: # File has a header but not the expected one
                     logger.warning(f"CSV file {output_csv} exists but has unexpected header: {header}. Will append with the correct header, possibly creating duplicates if resuming or mixing formats.")
                else: # File exists but is empty or has no header
                    logger.info(f"CSV file {output_csv} exists but is empty or lacks a header. Starting fresh.")
                    processed_paths = set() # Treat as fresh start
        except FileNotFoundError:
             logger.info(f"Output CSV {output_csv} not found. Starting fresh.")
             processed_paths = set()
        except StopIteration:
             logger.info(f"Output CSV {output_csv} exists but is empty. Starting fresh.")
             processed_paths = set()
        except Exception as e:
            logger.error(f"Could not read existing CSV {output_csv} to check for processed files: {e}. Processing all images, potentially creating duplicates.")
            processed_paths = set()

    to_process: List[str] = [img for img in all_images if img not in processed_paths]

    if not to_process:
        logger.info("All found images appear to be already processed according to the CSV file.")
        return
    logger.info(f"Starting gender, age, and dominant race inference for {len(to_process)} new images.")
    logger.info(f"Using up to {max_workers} parallel workers.")

    # --- Process Images in Parallel ---
    # Define the CSV header with inferred attributes
    csv_header = ["image_path", "gender", "age", "race"]

    # Determine if CSV header needs to be written
    # Ensure header is written if file is new, empty, or if we reset processed_paths due to read error/bad header
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0 or (os.path.exists(output_csv) and not processed_paths and os.path.getsize(output_csv) > 0)


    try:
        with open(output_csv, "a", newline="", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            if write_header:
                writer.writerow(csv_header)
                logger.info("CSV header written.")

            processed_count = 0
            total_to_process = len(to_process)
            print(f"🟢 Starting processing of {total_to_process} images...")

            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='DeepFaceWorker') as executor:
                futures = {executor.submit(analyze_image, img): img for img in to_process}

                for future in as_completed(futures):
                    # Unpack results - ignore race_scores (_ placeholder)
                    img_path_res, status, gender_res, age_res, dom_race_res, _ = future.result()

                    # Prepare row data: image_path, gender, age, race; missing attributes remain empty
                    row_data = [img_path_res, gender_res, age_res, dom_race_res]
                    writer.writerow(row_data)

                    processed_count += 1
                    print(f"✅ Processed {processed_count}/{total_to_process} images")

                    if processed_count % 50 == 0 or processed_count == total_to_process:
                        # Log progress at DEBUG level to avoid console spam
                        logger.debug(f"Progress: {processed_count}/{total_to_process} images processed.")

    except IOError as e:
         logger.critical(f"Could not write to output CSV file {output_csv}: {e}")
    except Exception as e:
         logger.critical(f"An unexpected error occurred during processing: {e}", exc_info=True)
    finally:
        print("🏁 Processing complete.")
        logger.info(f"Finished processing loop. Attempted {processed_count} images in this run.")
        logger.info(f"Results saved/appended to {output_csv}")
        logger.info(f"Detailed logs available in {log_file}")


# --- Command-Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel gender, age, and dominant race inference using DeepFace.", # Updated description
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input_dir",
        default="data",
        help="Directory containing images to process."
    )
    parser.add_argument(
        "-o", "--output_csv",
        default="image_attributes_summary.csv", # Changed default name slightly
        help="CSV file path to save or append results (gender, age, dominant race, status)."
    )
    parser.add_argument(
        "-l", "--log_file",
        default="infer_attributes.log",
        help="Log file path for detailed logs."
    )
    parser.add_argument(
        "-w", "--max_workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Maximum number of parallel worker threads."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging to the console."
    )

    args = parser.parse_args()

    if args.max_workers < 1:
        print("Warning: max_workers must be at least 1. Setting to 1.")
        args.max_workers = 1

    main(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        log_file=args.log_file,
        max_workers=args.max_workers,
        verbose=args.verbose
    )
