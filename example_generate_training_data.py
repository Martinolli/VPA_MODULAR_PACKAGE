"""
Example script demonstrating how to use the VPATrainingDataGenerator.
"""

import logging
import traceback

# Assuming modules are importable (e.g., project root added to PYTHONPATH or installed)
try:
    from vpa_modular.vpa_facade import VPAFacade
    from vpa_modular.vpa_training_data_generator import VPATrainingDataGenerator

    # --- Configuration ---
    TICKER = "MSFT"  # Example ticker
    START_DATE = "2025-01-01"
    END_DATE = "2025-03-01" # Shorter period for quicker example run
    PRIMARY_TIMEFRAME = "1d"
    SECONDARY_TIMEFRAMES = ["1h"] # Keep secondary TFs minimal for example speed
    OUTPUT_DIRECTORY = "./llm_training_data_output" # Specify output directory
    MIN_LOOKBACK = 50 # Minimum data points before starting generation

    # --- Setup Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("GenerateTrainingDataExample")

    # --- Main Execution ---
    logger.info("Starting VPA LLM Training Data Generation Example...")

    # 1. Initialize VPA Facade
    #    - Assumes default configuration is suitable or config file is found.
    #    - The facade provides access to the underlying VPA analysis logic.
    try:
        logger.info("Initializing VPAFacade...")
        # If your facade requires specific config, provide it here:
        # vpa_facade = VPAFacade(config_file="path/to/your/config.json")
        vpa_facade = VPAFacade()
        logger.info("VPAFacade initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize VPAFacade: {e}", exc_info=True)
        exit(1)

    # 2. Initialize Training Data Generator
    #    - Pass the initialized facade and the desired output directory.
    logger.info(f"Initializing VPATrainingDataGenerator (output dir: {OUTPUT_DIRECTORY})...")
    generator = VPATrainingDataGenerator(vpa_facade, output_dir=OUTPUT_DIRECTORY)
    logger.info("VPATrainingDataGenerator initialized successfully.")

    # 3. Run the Generation Process
    #    - Call generate_training_data with the defined parameters.
    #    - This will load data, iterate, analyze point-in-time (or mock), 
    #      format, and save to a JSONL file in the output directory.
    logger.info(f"Starting data generation for {TICKER} from {START_DATE} to {END_DATE}...")
    try:
        generator.generate_training_data(
            ticker=TICKER,
            start_date=START_DATE,
            end_date=END_DATE,
            primary_timeframe=PRIMARY_TIMEFRAME,
            other_timeframes=SECONDARY_TIMEFRAMES,
            min_lookback=MIN_LOOKBACK
        )
        logger.info(f"Data generation process completed for {TICKER}.")
        logger.info(f"Check the file 	\"{OUTPUT_DIRECTORY}/{TICKER}_vpa_training_data.jsonl\"")

    except Exception as e:
        logger.error(f"An error occurred during training data generation: {e}", exc_info=True)

except ImportError as e:
    print(f"Import Error: {e}. Make sure the VPA modules are installed or accessible in PYTHONPATH.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    traceback.print_exc()


