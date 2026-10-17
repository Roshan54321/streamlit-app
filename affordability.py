"""Affordability maths used by the dashboard. Kept separate from app.py so it can be unit-tested."""


def monthly_repayment(loan: float, annual_rate_pct: float, years: int) -> float:
    """Standard amortised monthly repayment. A zero rate means equal instalments."""
    if loan < 0 or years <= 0:
        raise ValueError("loan must be >= 0 and years > 0")
    r, n = annual_rate_pct / 100 / 12, years * 12
    return loan / n if r == 0 else loan * r / (1 - (1 + r) ** -n)


def rti(price: float, deposit_pct: float, rate_pct: float, years: int, income: float) -> float:
    """Repayment-to-income: yearly mortgage repayments as a % of gross household income."""
    if income <= 0:
        raise ValueError("income must be > 0")
    loan = price * (1 - deposit_pct / 100)
    return monthly_repayment(loan, rate_pct, years) * 12 / income * 100
