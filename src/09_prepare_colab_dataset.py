"""Prepare an all-filing, 12-month amendment-proxy dataset from local SEC FSDS files.

This reads local SUB/NUM files only. It makes no SEC requests and does not label
confirmed restatements. GPU fitting is performed by the Colab notebook.
"""

from pathlib import Path
import numpy as np
import pandas as pd


METADATA_COLUMNS = ["adsh", "cik", "form", "period", "filed", "fp", "sic", "afs", "wksi"]
BALANCE_TAGS = {
    "Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent",
    "StockholdersEquity", "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "LongTermDebtCurrent", "LongTermDebtNoncurrent", "AccountsReceivableNetCurrent",
    "InventoryNet", "PropertyPlantAndEquipmentNet", "AccountsPayableCurrent",
    "RetainedEarningsAccumulatedDeficit",
    "Goodwill", "IntangibleAssetsNetExcludingGoodwill", "ShortTermBorrowings",
    "LongTermDebt", "OperatingLeaseLiabilityCurrent", "OperatingLeaseLiabilityNoncurrent",
    "ContractWithCustomerAssetCurrent", "ContractWithCustomerLiabilityCurrent",
    "DeferredRevenueCurrent", "DeferredIncomeTaxAssetsNet", "DeferredIncomeTaxLiabilitiesNet",
    "AvailableForSaleSecuritiesCurrent", "MarketableSecuritiesCurrent",
    "OtherAssetsCurrent", "OtherLiabilitiesCurrent", "OtherAssetsNoncurrent",
    "OtherLiabilitiesNoncurrent", "AccountsReceivableNetCurrent",
    "AllowanceForDoubtfulAccountsReceivableCurrent", "NotesPayableCurrent",
}
FLOW_TAGS = {
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
    "NetIncomeLoss", "OperatingIncomeLoss", "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment", "GrossProfit", "CostOfRevenue",
    "ResearchAndDevelopmentExpense", "SellingGeneralAndAdministrativeExpense",
    "IncomeTaxExpenseBenefit", "InterestExpenseNonOperating",
    "DepreciationDepletionAndAmortization", "OperatingExpenses", "ProfitLoss",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "ShareBasedCompensation", "PaymentsToAcquireBusinessesNetOfCashAcquired",
    "PaymentsOfDividendsCommonStock", "PaymentsOfDividends",
    "PaymentsForRepurchaseOfCommonStock", "StockBasedCompensation",
    "ProvisionForDoubtfulAccounts", "ImpairmentOfGoodwill",
    "RestructuringCharges", "GainLossOnSaleOfAssets",
}
NUMERIC_TAGS = BALANCE_TAGS | FLOW_TAGS


def load_submissions(raw_dir):
    parts = []
    for path in sorted(raw_dir.glob("*/sub.txt")):
        frame = pd.read_csv(path, sep="\t", dtype=str, usecols=lambda c: c in METADATA_COLUMNS)
        parts.append(frame[frame.form.isin(["10-K", "10-Q", "10-K/A", "10-Q/A"])])
    if not parts:
        raise FileNotFoundError(f"No quarterly sub.txt files found under {raw_dir}")
    df = pd.concat(parts, ignore_index=True).drop_duplicates("adsh")
    df["filed_date"] = pd.to_datetime(df.filed, format="%Y%m%d", errors="coerce")
    df["period_date"] = pd.to_datetime(df.period, format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["cik", "period_date", "filed_date"])
    return df


def make_target(submissions):
    originals = submissions[submissions.form.isin(["10-K", "10-Q"])].copy()
    amendments = submissions[submissions.form.isin(["10-K/A", "10-Q/A"])].copy()
    amendments["base_form"] = amendments.form.str.removesuffix("/A")
    pairs = originals[["adsh", "cik", "period", "form", "filed_date"]].merge(
        amendments[["cik", "period", "base_form", "filed_date"]],
        left_on=["cik", "period", "form"],
        right_on=["cik", "period", "base_form"],
        suffixes=("_original", "_amendment"),
    )
    pairs["days_to_amendment"] = (
        pairs.filed_date_amendment - pairs.filed_date_original
    ).dt.days
    pairs = pairs[pairs.days_to_amendment.between(1, 365)]
    return originals, pairs.groupby("adsh", as_index=False).days_to_amendment.min()


def quarter_count(row):
    fp = str(row.fp).upper()
    if row.form == "10-K":
        return 4
    return {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}.get(fp, 0)


def extract_numeric_features(raw_dir, originals):
    original_ids = set(originals.adsh)
    filing_meta = originals.set_index("adsh")[["period", "form", "fp"]]
    parts = []
    selected_cols = ["adsh", "tag", "ddate", "qtrs", "uom", "segments", "coreg", "value"]
    quarter_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    for qdir in quarter_dirs:
        num_path = qdir / "num.txt"
        sub_path = qdir / "sub.txt"
        if not num_path.exists() or not sub_path.exists():
            continue
        q_sub = pd.read_csv(sub_path, sep="\t", dtype=str, usecols=lambda c: c in METADATA_COLUMNS)
        q_ids = set(q_sub.loc[q_sub.form.isin(["10-K", "10-Q"]), "adsh"]) & original_ids
        if not q_ids:
            continue
        q_meta = filing_meta.loc[list(q_ids)].reset_index()
        q_meta["flow_qtrs"] = q_meta.apply(quarter_count, axis=1)
        q_features = []
        available = set(pd.read_csv(num_path, sep="\t", nrows=0).columns)
        usecols = [column for column in selected_cols if column in available]
        for chunk in pd.read_csv(num_path, sep="\t", dtype=str, usecols=usecols, chunksize=500_000):
            for optional in ("segments", "coreg"):
                if optional not in chunk:
                    chunk[optional] = ""
            chunk = chunk[chunk.adsh.isin(q_ids) & chunk.tag.isin(NUMERIC_TAGS) & chunk.uom.eq("USD")]
            if chunk.empty:
                continue
            chunk = chunk[
                chunk.segments.isna() | chunk.segments.astype(str).str.strip().eq("")
            ]
            chunk = chunk[
                chunk.coreg.isna() | chunk.coreg.astype(str).str.strip().eq("")
            ]
            if chunk.empty:
                continue
            chunk = chunk.merge(q_meta[["adsh", "period", "flow_qtrs"]], on="adsh", how="inner")
            chunk = chunk[chunk.ddate.eq(chunk.period)]
            is_balance = chunk.tag.isin(BALANCE_TAGS)
            chunk = chunk[(is_balance & chunk.qtrs.eq("0")) | (~is_balance & chunk.qtrs.eq(chunk.flow_qtrs.astype(str)))]
            if chunk.empty:
                continue
            chunk["value"] = pd.to_numeric(chunk.value, errors="coerce")
            chunk = chunk.dropna(subset=["value"])
            if not chunk.empty:
                q_features.append(chunk.groupby(["adsh", "tag"], as_index=False).value.median())
        if q_features:
            long = pd.concat(q_features, ignore_index=True)
            wide = long.groupby(["adsh", "tag"], as_index=False).value.median().pivot(
                index="adsh", columns="tag", values="value"
            )
            wide.columns = [f"xbrl_{c}" for c in wide.columns]
            parts.append(wide.reset_index())
        print(f"Read numeric facts: {qdir.name}")
    if not parts:
        raise ValueError("No matching as-filed numeric facts found in NUM files")
    return pd.concat(parts, ignore_index=True).drop_duplicates("adsh", keep="last")


def add_issuer_history(df, submissions):
    """Add history known strictly before each original filing's submission day."""
    amendments = submissions[submissions.form.isin(["10-K/A", "10-Q/A"])][["cik", "filed_date"]]
    original_history = submissions[submissions.form.isin(["10-K", "10-Q"])][
        ["cik", "filed_date", "period_date"]
    ].copy()
    original_history["delay"] = (original_history.filed_date - original_history.period_date).dt.days.clip(0, 730)
    amendment_groups = {
        cik: np.sort(group.filed_date.to_numpy(dtype="datetime64[D]"))
        for cik, group in amendments.groupby("cik", sort=False)
    }
    original_groups = {
        cik: group.sort_values("filed_date") for cik, group in original_history.groupby("cik", sort=False)
    }
    df["prior_amendment_count"] = 0
    df["prior_amendments_365d"] = 0
    df["days_since_prior_amendment"] = np.nan
    df["prior_original_count"] = 0
    df["days_since_prior_original"] = np.nan
    df["prior_mean_filing_delay"] = np.nan
    for cik, rows in df.groupby("cik", sort=False):
        target_dates = rows.filed_date.to_numpy(dtype="datetime64[D]")
        amendment_dates = amendment_groups.get(cik, np.array([], dtype="datetime64[D]"))
        position = np.searchsorted(amendment_dates, target_dates, side="left")
        year_ago = np.searchsorted(amendment_dates, target_dates - np.timedelta64(365, "D"), side="left")
        df.loc[rows.index, "prior_amendment_count"] = position
        df.loc[rows.index, "prior_amendments_365d"] = position - year_ago
        has_amendment = position > 0
        if has_amendment.any():
            indices = rows.index[has_amendment]
            df.loc[indices, "days_since_prior_amendment"] = (
                target_dates[has_amendment] - amendment_dates[position[has_amendment] - 1]
            ).astype("timedelta64[D]").astype(int)
        original_rows = original_groups[cik]
        original_dates = original_rows.filed_date.to_numpy(dtype="datetime64[D]")
        prior_position = np.searchsorted(original_dates, target_dates, side="left")
        df.loc[rows.index, "prior_original_count"] = prior_position
        has_original = prior_position > 0
        if has_original.any():
            indices = rows.index[has_original]
            df.loc[indices, "days_since_prior_original"] = (
                target_dates[has_original] - original_dates[prior_position[has_original] - 1]
            ).astype("timedelta64[D]").astype(int)
            delays = original_rows.delay.to_numpy(dtype=float)
            count = np.arange(1, len(delays) + 1)
            prefix_mean = np.nancumsum(delays) / count
            df.loc[indices, "prior_mean_filing_delay"] = prefix_mean[prior_position[has_original] - 1]
    return df


def add_features(df):
    raw_amounts = [c for c in df.columns if c.startswith("xbrl_")]
    # Build financial ratios from untransformed dollar values. Applying ratios
    # to the log-transformed facts (as the previous version did) is invalid.
    for col in raw_amounts:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def fact(*names):
        candidates = [df.get(f"xbrl_{name}", pd.Series(np.nan, index=df.index)) for name in names]
        return pd.concat(candidates, axis=1).bfill(axis=1).iloc[:, 0]

    def ratio_values(numerator, denominator):
        a = pd.to_numeric(numerator, errors="coerce")
        b = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
        return (a / b).replace([np.inf, -np.inf], np.nan).clip(-100, 100)

    assets = fact("Assets")
    current_assets = fact("AssetsCurrent")
    current_liabilities = fact("LiabilitiesCurrent")
    cash = fact("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents")
    revenue = fact("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")
    net_income = fact("NetIncomeLoss", "ProfitLoss")
    operating_income = fact("OperatingIncomeLoss")
    cash_flow = fact("NetCashProvidedByUsedInOperatingActivities")
    debt = fact("LongTermDebt")
    debt = debt.fillna(fact("LongTermDebtCurrent").fillna(0) + fact("LongTermDebtNoncurrent").fillna(0))
    receivables = fact("AccountsReceivableNetCurrent")
    inventory = fact("InventoryNet")
    equity = fact("StockholdersEquity")
    gross_profit = fact("GrossProfit")
    capex = fact("PaymentsToAcquirePropertyPlantAndEquipment")
    retained_earnings = fact("RetainedEarningsAccumulatedDeficit")

    derived = {
        "liabilities_to_assets": ratio_values(fact("Liabilities"), assets),
        "current_assets_to_current_liabilities": ratio_values(current_assets, current_liabilities),
        "working_capital_to_assets": ratio_values(current_assets - current_liabilities, assets),
        "cash_to_assets": ratio_values(cash, assets),
        "receivables_to_revenue": ratio_values(receivables, revenue),
        "inventory_to_revenue": ratio_values(inventory, revenue),
        "gross_margin": ratio_values(gross_profit, revenue),
        "operating_margin": ratio_values(operating_income, revenue),
        "net_margin": ratio_values(net_income, revenue),
        "net_income_to_assets": ratio_values(net_income, assets),
        "operating_cashflow_to_assets": ratio_values(cash_flow, assets),
        "operating_cashflow_to_net_income": ratio_values(cash_flow, net_income),
        "accruals_to_assets": ratio_values(net_income - cash_flow, assets),
        "debt_to_assets": ratio_values(debt, assets),
        "liabilities_to_equity": ratio_values(fact("Liabilities"), equity),
        "retained_earnings_to_assets": ratio_values(retained_earnings, assets),
        "capex_to_operating_cashflow": ratio_values(capex, cash_flow.abs()),
        "revenue_to_assets": ratio_values(revenue, assets),
    }
    for name, values in derived.items():
        df[name] = values

    # Signed-log dollar amounts are numerically stable; ratio features remain
    # true unitless financial ratios.
    for col in raw_amounts:
        values = df[col]
        df[col] = np.sign(values) * np.log1p(np.abs(values))

    # Compare each filing with the prior same-form, same-fiscal-period filing.
    # The lag is based only on earlier submissions and periods.
    history_columns = raw_amounts + list(derived)
    group_columns = ["cik", "form", "fp"]
    history = df.sort_values(["filed_date", "period_date", "adsh"]).drop_duplicates(
        group_columns + ["period_date"], keep="first"
    ).copy()
    grouped = history.groupby(group_columns, sort=False)
    prior_period = grouped["period_date"].shift(1)
    prior_filed = grouped["filed_date"].shift(1)
    has_prior = (
        prior_period.notna()
        & prior_filed.lt(history.filed_date)
        & prior_period.lt(history.period_date)
    )
    history["has_prior_comparable"] = has_prior.astype("int8")
    history["days_since_comparable_filing"] = (history.filed_date - prior_filed).dt.days.where(has_prior)
    for col in history_columns:
        prior_value = grouped[col].shift(1)
        history[f"delta_{col}"] = (history[col] - prior_value).where(has_prior)
    lag_columns = ["adsh", "has_prior_comparable", "days_since_comparable_filing"] + [
        f"delta_{col}" for col in history_columns
    ]
    df = df.merge(history[lag_columns], on="adsh", how="left", validate="one_to_one")
    df["has_prior_comparable"] = df.has_prior_comparable.fillna(0).astype("int8")

    df["filing_month"] = df.filed_date.dt.month
    # Available when the filing is submitted; this captures reporting delay.
    df["filing_delay_days"] = (df.filed_date - df.period_date).dt.days.clip(lower=0, upper=730)
    df["period_month"] = df.period_date.dt.month
    df["sic2"] = df.sic.fillna("").astype(str).str.extract(r"^(\d{2})", expand=False).fillna("unknown")
    df["form"] = df.form.astype(str)
    df["fp"] = df.fp.fillna("unknown").astype(str)
    df["afs"] = df.afs.fillna("unknown").astype(str)
    df["feature_schema_version"] = 4
    return df


def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    out_path = root / "data" / "colab_amendment_proxy.csv.gz"
    quarters = sorted(p.name for p in raw_dir.iterdir() if p.is_dir() and len(p.name) == 6 and p.name[4] == "Q")
    for quarter in quarters:
        missing = {"sub.txt", "num.txt"} - {p.name for p in (raw_dir / quarter).iterdir()}
        if missing:
            raise FileNotFoundError(f"Incomplete SEC quarter {quarter}: {sorted(missing)}")
    if quarters:
        start = int(quarters[0][:4]) * 4 + int(quarters[0][-1]) - 1
        end = int(quarters[-1][:4]) * 4 + int(quarters[-1][-1]) - 1
        expected = {f"{n // 4}Q{n % 4 + 1}" for n in range(start, end + 1)}
        if expected - set(quarters):
            raise ValueError(f"Missing SEC quarters: {sorted(expected - set(quarters))}")
    submissions = load_submissions(raw_dir)
    # A complete SEC quarterly archive covers its calendar-quarter end even if
    # no selected 10-K/10-Q happened to be submitted on that exact day.
    cutoff = pd.Period(quarters[-1], freq="Q").end_time.normalize()
    originals, event_days = make_target(submissions)
    # Include only filings with a complete 365-day label window in the local SEC files.
    originals = originals[originals.filed_date <= cutoff - pd.Timedelta(days=365)].copy()
    originals = originals.merge(event_days, on="adsh", how="left")
    originals["y_amendment_proxy"] = originals.days_to_amendment.notna().astype("int8")
    print(f"SEC submission cutoff: {cutoff.date()}")
    print(f"Eligible original filings: {len(originals):,}; positives: {int(originals.y_amendment_proxy.sum()):,}")

    numeric = extract_numeric_features(raw_dir, originals)
    data = originals.merge(numeric, on="adsh", how="left", validate="one_to_one")
    data = add_features(data)
    data = add_issuer_history(data, submissions)
    data["filed_year"] = data.filed_date.dt.year
    data["label_observation_cutoff"] = cutoff.strftime("%Y-%m-%d")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False, compression="gzip")
    print(f"Saved {len(data):,} rows x {len(data.columns):,} columns to {out_path}")
    print("Target is an amendment within 365 days, not a confirmed financial restatement.")


if __name__ == "__main__":
    main()
