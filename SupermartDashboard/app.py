
from flask import Flask, render_template, request
import pandas as pd
from pathlib import Path

app = Flask(__name__)

DATA_FILE = Path(__file__).parent / (
    "Supermart Grocery Sales - Retail Analytics Dataset.csv"
)


def load_data():
    df = pd.read_csv(DATA_FILE)

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    for col in ["Sales", "Profit", "Discount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["Order Date", "Sales", "Profit"]
    ).copy()

    df["Month"] = df["Order Date"].dt.strftime("%Y-%m")
    return df


DATA = load_data()


def money(value):
    return f"₹{value:,.2f}"


def make_bars(series):
    if series.empty:
        return []

    maximum = max(float(series.max()), 1)
    bars = []

    for label, value in series.sort_values(
        ascending=False
    ).items():
        bars.append({
            "label": str(label),
            "value": money(value),
            "width": max(2, float(value) / maximum * 100)
        })

    return bars


@app.route("/")
def dashboard():
    categories = sorted(DATA["Category"].dropna().unique())
    regions = sorted(DATA["Region"].dropna().unique())
    cities = sorted(DATA["City"].dropna().unique())

    start_default = DATA["Order Date"].min().strftime("%Y-%m-%d")
    end_default = DATA["Order Date"].max().strftime("%Y-%m-%d")

    start = request.args.get("start", start_default)
    end = request.args.get("end", end_default)

    selected_categories = request.args.getlist("category")
    selected_regions = request.args.getlist("region")
    selected_cities = request.args.getlist("city")

    if not request.args:
        selected_categories = list(categories)
        selected_regions = list(regions)
        selected_cities = list(cities)

    df = DATA.copy()

    try:
        df = df[
            (df["Order Date"] >= pd.Timestamp(start)) &
            (df["Order Date"] < (
                pd.Timestamp(end) + pd.Timedelta(days=1)
            ))
        ]
    except (ValueError, TypeError):
        start = start_default
        end = end_default

    df = df[
        df["Category"].isin(selected_categories) &
        df["Region"].isin(selected_regions) &
        df["City"].isin(selected_cities)
    ].copy()

    sales = float(df["Sales"].sum())
    profit = float(df["Profit"].sum())
    margin = profit / sales * 100 if sales else 0
    average_discount = (
        float(df["Discount"].mean() * 100) if not df.empty else 0
    )

    category_sales = df.groupby("Category")["Sales"].sum()
    region_sales = df.groupby("Region")["Sales"].sum()

    monthly_sales = (
        df.groupby("Month")["Sales"]
        .sum()
        .sort_index()
        .tail(24)
    )

    monthly_bars = make_bars(monthly_sales)

    products_df = (
        df.groupby(["Category", "Sub Category"], as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "nunique")
        )
        .sort_values("Sales", ascending=False)
        .head(20)
    )

    products = products_df.to_dict("records")

    discount_df = df.copy()
    discount_df["Discount Band"] = pd.cut(
        discount_df["Discount"],
        bins=[-0.001, 0.10, 0.20, 0.30, 0.40, 1],
        labels=["0-10%", "10-20%", "20-30%", "30-40%", "40%+"]
    )

    discount_profit = (
        discount_df.groupby("Discount Band", observed=False)["Profit"]
        .sum()
    )

    if len(df) > 1:
        correlation = df["Discount"].corr(df["Profit"])
        if pd.isna(correlation):
            correlation = 0
    else:
        correlation = 0

    geography_df = (
        df.groupby(["State", "City", "Region"], as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .sort_values("Sales", ascending=False)
        .head(25)
    )

    geography = geography_df.to_dict("records")

    return render_template(
        "index.html",
        row_count=len(df),
        sales=money(sales),
        profit=money(profit),
        margin=f"{margin:.2f}%",
        average_discount=f"{average_discount:.2f}%",
        order_count=f"{df['Order ID'].nunique():,}",
        category_bars=make_bars(category_sales),
        region_bars=make_bars(region_sales),
        monthly_bars=monthly_bars,
        discount_bars=make_bars(discount_profit),
        products=products,
        geography=geography,
        correlation=f"{correlation:.4f}",
        categories=categories,
        regions=regions,
        cities=cities,
        selected_categories=selected_categories,
        selected_regions=selected_regions,
        selected_cities=selected_cities,
        start=start,
        end=end,
        empty=df.empty,
        state_count=df["State"].nunique() if not df.empty else 0
    )


if __name__ == "__main__":
    app.run(debug=True)
