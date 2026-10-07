def credit_band_lookup(income: int) -> str:
    if income >= 80000:
        return "good"
    if income >= 65000:
        return "fair"
    return "poor"
