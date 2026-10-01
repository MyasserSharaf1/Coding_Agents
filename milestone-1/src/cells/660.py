import zipfile

ZIP_PATH = os.path.join(BASE_DIR, "m1_%s_results.zip" % M1_PART)
n_files = 0
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(M1_OUTPUT_DIR):
        for fn in sorted(files):
            full = os.path.join(root, fn)
            z.write(full, os.path.relpath(full, BASE_DIR))
            n_files += 1

print("Packed %d files -> %s  (%.1f MB)"
      % (n_files, os.path.abspath(ZIP_PATH), os.path.getsize(ZIP_PATH) / 1e6))
with zipfile.ZipFile(ZIP_PATH) as z:
    for name in z.namelist():
        print("   ", name)

if globals().get("IS_COLAB"):
    from google.colab import files
    files.download(ZIP_PATH)
else:
    try:
        from IPython.display import FileLink, display
        print("\nClick to download:")
        display(FileLink(os.path.relpath(ZIP_PATH)))
    except Exception:
        print("\nDownload it from:", os.path.abspath(ZIP_PATH))
    if ON_KAGGLE:
        print("On Kaggle it is also listed in the Output panel (right sidebar) -> %s" % os.path.basename(ZIP_PATH))