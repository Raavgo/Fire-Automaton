def validateText(text):
    try:
        return float(text)
    except ValueError:
        return 0