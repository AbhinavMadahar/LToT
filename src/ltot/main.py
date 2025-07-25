#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lateral Tree-of-Thoughts (LToT)

This script runs experiments for the Lateral Tree-of-Thoughts reasoning architecture.
It reads an experimental configuration file, runs the experiment accordingly,
and saves the results in a corresponding specified output directory.

Vanilla ToT is also supported, and can be used by setting the lateral width to zero.
"""


import argparse
import logging


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Lateral Tree-of-Thoughts (LToT) Application')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--configuration',
                        type=str,
                        required=True,
                        help='Path to the experimental configuration file')
    parser.add_argument('--results',
                        type=str,
                        required=True,
                        help='Directory to write the results to (use "-" to discard results). '
                             'A subdirectory with the name `<name-of-experiment>--YYYY-MM-DD-HH-MM-SS--UID` '
                             'will be created to store the results of the experimental run. Here, '
                             '`<name-of-experiment>` is the name of the experiment as specified in the '
                             'configuration file, `YYYY-MM-DD-HH-MM-SS` is the timestamp of the run (start time) at UTC+0, '
                             'and `UID` is a unique identifier for the run.')
    parser.add_argument('--results-subdirectory',
                        type=str,
                        default=None,
                        help='For the rare cases where you want to specify the exact subdirectory name '
                             'for the results, you can do so here. Will error if the subdirectory already exists.')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    # Main application logic
    logger.info('Starting experiment...')
    raise NotImplementedError('Experiment logic is not implemented yet.')
