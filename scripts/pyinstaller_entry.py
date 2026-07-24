import multiprocessing

from olist_code.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
