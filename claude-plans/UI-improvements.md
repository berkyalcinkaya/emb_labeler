To do to improve the UI:
1. Remove all references to best depths. This should also propagate to data storage. We no longer need best depth labels.
2. Remove the Auto RLI toggle.
3. OCR, Pred, and RLI status is not visible enough. I want to revert back to a progress bar. Please put a progress bar on the far right panel for each, with errors surfacing somehow in the progress bar for Pred if a time point model has not been loaded by AWS.