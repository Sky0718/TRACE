from trace_robustness.batch import main

if (__name__ == "__main__"):
    import multiprocessing

    multiprocessing.freeze_support()
    main()
