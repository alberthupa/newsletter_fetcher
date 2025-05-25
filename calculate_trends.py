import os
from collections import defaultdict, Counter
import datetime
from collections import Counter, defaultdict
import json  # For potentially loading mock data or saving results

from src.vectors.cosmos_client import SimpleCosmosClient


COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"
PARTITION_KEY_PATH = "/id"

# --- Configuration ---
DEBUG_PRINT = False  # Set to False to suppress detailed statistical prints

KEYWORD_FIELDS = [
    "keywords",
    "companies",
    "model_name",
    "model_architecture",
    "detailed_model_version",
    "ai_tools",
    "infrastructure",
    "ml_techniques",
]

# Define time windows in days (relative to 'today', which is the execution day)
# We'll typically analyze data *up to* 'yesterday'.
TIME_WINDOWS = {
    "daily": 2,  # Data from today and yesterday (2 days)
    "weekly": 7,  # Data from the last 7 days (yesterday to 7 days ago)
    "monthly": 30,  # Data from the last 30 days
    "quarterly": 90,  # Data from the last 90 days
}

SIGNIFICANT_INCREASE_THRESHOLD = 2.0  # Factor by which current count must exceed baseline average to be "significant"
TOP_N_FOR_PERIOD_SUMMARY = 5  # Number of top items to show for each period's summary
TOP_N_FOR_EMERGING_TREND_DETAILS = (
    10  # Max number of items to show in flattened emerging trend sections
)


cosmos_client = SimpleCosmosClient(
    connection_string=COSMOS_CONNECTION_STRING,
    database_name=DATABASE_NAME,
    partition_key_path=PARTITION_KEY_PATH,
)

cosmos_client.connect()
container = cosmos_client.database_client.get_container_client("knowledge-pieces")
reports_container = cosmos_client.database_client.get_container_client(
    "knowledge-reports"
)


def get_items_for_date_range(start_date_str, end_date_str):
    query = f"SELECT c.id, c.chunk_date, {', '.join(['c.' + f for f in KEYWORD_FIELDS])} FROM c WHERE c.chunk_date >= '{start_date_str}' AND c.chunk_date <= '{end_date_str}'"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    print(f"Found {len(items)} items.")
    return items


def get_latest_date_from_items(items):
    """
    Extract the latest chunk_date from a list of items.
    Returns the date as a string in 'YYYY-MM-DD' format.
    """
    if not items:
        return None

    latest_date = None
    for item in items:
        if "chunk_date" in item:
            item_date = item["chunk_date"]
            if latest_date is None or item_date > latest_date:
                latest_date = item_date

    return latest_date


def save_trend_report(report_content, trend_flag, report_date):
    """
    Save a trend report to the knowledge-reports container.

    Args:
        report_content (str): The content of the trend report
        trend_flag (str): The type of trend - "daily", "weekly", or "monthly"
        report_date (str): The date of the report in 'YYYY-MM-DD' format
    """
    if not report_content or not trend_flag or not report_date:
        print(
            f"Warning: Cannot save report with missing data - trend_flag: {trend_flag}, report_date: {report_date}, content length: {len(report_content) if report_content else 0}"
        )
        return

    # Create a unique ID for the report
    report_id = f"{trend_flag}-{report_date}"

    report_item = {
        "id": report_id,
        "trend_report": report_content,
        "trend_flag": trend_flag,
        "report_date": report_date,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }

    try:
        reports_container.upsert_item(body=report_item)
        print(f"Successfully saved {trend_flag} trend report for {report_date}")
    except Exception as e:
        print(f"Error saving {trend_flag} trend report: {e}")


def get_headlines_for_keyword(
    keyword_text,
    keyword_field_name,
    start_date_str,
    end_date_str,
    container_client,
    limit=3,
):
    """
    Fetches up to 'limit' distinct headlines for a given keyword and field within a date range.
    """
    if not keyword_text or not keyword_field_name:
        return []

    # Escape single quotes in keyword_text for SQL query
    escaped_keyword_text = keyword_text.replace("'", "''")

    # Construct the query
    # Ensure the keyword_field_name is a valid field and not arbitrary input
    if keyword_field_name not in KEYWORD_FIELDS:
        print(
            f"Warning: Invalid keyword_field_name '{keyword_field_name}' provided to get_headlines_for_keyword."
        )
        return []

    query = (
        f"SELECT DISTINCT TOP {limit} c.headline "
        f"FROM c "
        f"WHERE c.chunk_date >= '{start_date_str}' AND c.chunk_date <= '{end_date_str}' "
        f"AND ARRAY_CONTAINS(c.{keyword_field_name}, '{escaped_keyword_text}', true)"  # Using true for case-insensitive check if desired, or remove for exact match
    )
    # print(f"Headline query: {query}") # For debugging

    try:
        query_results = list(
            container_client.query_items(query=query, enable_cross_partition_query=True)
        )
        headlines = [
            item["headline"]
            for item in query_results
            if "headline" in item and item["headline"]
        ]
        return headlines
    except Exception as e:
        print(
            f"Error fetching headlines for '{keyword_text}' in field '{keyword_field_name}': {e}"
        )
        return []


# --- Date Utilities ---
def get_date_ranges(today_date):
    """
    Calculates the start and end dates for each time window relative to today_date.
    For daily window, includes today and yesterday.
    For other windows, analysis is done up to 'yesterday'.
    Returns a dictionary: {"window_name": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
    """
    ranges = {}
    yesterday = today_date - datetime.timedelta(days=1)

    for window_name, days_delta in TIME_WINDOWS.items():
        if window_name == "daily":
            # Daily window includes today and yesterday
            start_date = yesterday
            end_date = today_date
        else:
            # Other windows are calculated from yesterday backwards
            start_date = yesterday - datetime.timedelta(days=days_delta - 1)
            end_date = yesterday

        ranges[window_name] = {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
        }
    return ranges


def get_date_ranges_with_fallback(today_date, window_name, fallback_days=0):
    """
    Get date ranges with a fallback option for the daily window.
    If fallback_days > 0, it shifts the window backwards by that many days.
    """
    ranges = {}

    if window_name == "daily":
        if fallback_days == 0:
            # Normal case: today and yesterday
            start_date = today_date - datetime.timedelta(days=1)
            end_date = today_date
        else:
            # Fallback case: shift backwards by fallback_days
            end_date = today_date - datetime.timedelta(days=fallback_days)
            start_date = end_date - datetime.timedelta(days=1)
    else:
        # For non-daily windows, use the standard logic
        yesterday = today_date - datetime.timedelta(days=1)
        days_delta = TIME_WINDOWS[window_name]
        start_date = yesterday - datetime.timedelta(days=days_delta - 1)
        end_date = yesterday

    ranges[window_name] = {
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
    }
    return ranges


# --- Data Processing and Aggregation ---
def aggregate_keyword_counts(items):
    """
    Aggregates counts of each keyword from the provided items.
    Returns a dictionary: {"field_name": Counter_object}
    """
    aggregated_counts = defaultdict(Counter)
    if not items:
        return aggregated_counts

    for item in items:
        for field in KEYWORD_FIELDS:
            if field in item and isinstance(item[field], list):
                for keyword in item[field]:
                    if keyword:
                        aggregated_counts[field][keyword.strip()] += 1
    return aggregated_counts


# --- Trend Analysis ---
def identify_top_n_keywords(period_counts, n=10):
    """
    Identifies the top N keywords for each field in a given period's counts.
    Returns: {"field_name": [("keyword", count), ...]}
    """
    top_keywords = {}
    for field, counts in period_counts.items():
        top_keywords[field] = counts.most_common(n)
    return top_keywords


def compare_trends(
    current_period_counts,
    adjusted_previous_period_average_counts,
    significant_increase_threshold=2.0,
):
    """
    Compares current period counts to adjusted average counts from a previous baseline.
    Returns:
        "newly_emerging": {"field_name": [("keyword", current_count), ...]}
        "significantly_increased": {"field_name": [("keyword", current_count, prev_avg_count, increase_factor), ...]}
    """
    newly_emerging = defaultdict(list)
    significantly_increased = defaultdict(list)

    for field in KEYWORD_FIELDS:
        current_field_counts = current_period_counts.get(field, Counter())

        for keyword, current_count in current_field_counts.items():
            previous_avg_count = adjusted_previous_period_average_counts.get(
                field, Counter()
            ).get(keyword, 0)

            if previous_avg_count < 0.01 and current_count > 0:
                newly_emerging[field].append((keyword, current_count))
            elif (
                previous_avg_count > 0
                and current_count / previous_avg_count >= significant_increase_threshold
            ):
                increase_factor = current_count / previous_avg_count
                significantly_increased[field].append(
                    (
                        keyword,
                        current_count,
                        round(previous_avg_count, 2),
                        round(increase_factor, 2),
                    )
                )

        newly_emerging[field].sort(key=lambda x: x[1], reverse=True)
        significantly_increased[field].sort(
            key=lambda x: (x[3], x[1]), reverse=True
        )  # Sort by factor, then current_count

    return {
        "newly_emerging": newly_emerging,
        "significantly_increased": significantly_increased,
    }


def calculate_adjusted_baseline_counts(
    current_period_counts,
    baseline_period_counts,
    days_current_period,
    days_baseline_period,
):
    """
    Calculates the average keyword counts for a period equivalent to the 'current_period'
    derived from the 'baseline_period', excluding the contribution of the 'current_period' itself.
    """
    adjusted_avg_counts = defaultdict(Counter)

    if days_baseline_period <= days_current_period:
        # This case should be handled before calling, but as a safeguard
        return adjusted_avg_counts

    days_in_remainder_of_baseline = days_baseline_period - days_current_period
    if days_in_remainder_of_baseline <= 0:
        return adjusted_avg_counts  # No remainder to calculate average from

    num_equivalent_current_periods_in_remainder = (
        days_in_remainder_of_baseline / days_current_period
    )
    if (
        num_equivalent_current_periods_in_remainder <= 0
    ):  # Should not happen if days_in_remainder > 0
        return adjusted_avg_counts

    for field in KEYWORD_FIELDS:
        all_keywords_in_field_baseline = baseline_period_counts.get(
            field, Counter()
        ).keys()
        for keyword in all_keywords_in_field_baseline:
            current_kw_count_in_current_period = current_period_counts.get(
                field, Counter()
            ).get(keyword, 0)
            baseline_kw_count_overall = baseline_period_counts.get(
                field, Counter()
            ).get(keyword, 0)

            # Count of keyword in the baseline period EXCLUDING the current period's contribution
            count_in_remainder = max(
                0, baseline_kw_count_overall - current_kw_count_in_current_period
            )

            avg_kw_count_in_equivalent_prior_period = (
                count_in_remainder / num_equivalent_current_periods_in_remainder
            )
            adjusted_avg_counts[field][keyword] = (
                avg_kw_count_in_equivalent_prior_period
            )

    return adjusted_avg_counts


# --- Helper for Report Formatting ---
def format_trend_report_section(
    title,
    keyword_data_list,
    item_type,  # "new" or "growing"
    date_ranges_for_windows,
    current_win_name,
    container_client,  # Passed as 'container' from main
    significant_increase_threshold_for_title=None,  # For "Rapidly Growing" title
    top_n_details=TOP_N_FOR_EMERGING_TREND_DETAILS,
    fetch_headlines_top_n=3,  # Number of top items to fetch headlines for
    debug_print_mode=DEBUG_PRINT,
):
    lines = []
    if item_type == "growing" and significant_increase_threshold_for_title is not None:
        lines.append(
            f"\n  {title} (Factor >= {significant_increase_threshold_for_title}x, vs. avg. of prior periods from baseline):"
        )
    else:
        lines.append(f"\n  {title}:")

    if not keyword_data_list:
        lines.append("    No keywords found for this category.")
        return lines

    report_start_date = date_ranges_for_windows[current_win_name]["start"]
    report_end_date = date_ranges_for_windows[current_win_name]["end"]

    for i, item in enumerate(keyword_data_list[:top_n_details]):
        base_info = (
            f"    - {item['keyword']} (from {item['field'].replace('_',' ').title()})"
        )
        if item_type == "new":
            if debug_print_mode:
                lines.append(f"{base_info} - Count: {item['count']}")
            else:
                lines.append(base_info)
        elif item_type == "growing":
            if debug_print_mode:
                lines.append(
                    f"{base_info} - Now: {item['current_count']}, Avg. Prior: {item['prev_avg']:.2f}, Factor: {item['factor']:.2f}x"
                )
            else:
                lines.append(base_info)

        if i < fetch_headlines_top_n:
            headlines = get_headlines_for_keyword(
                item["keyword"],
                item["field"],
                report_start_date,
                report_end_date,
                container_client,
                limit=3,  # Original logic fetched 3 headlines
            )
            if headlines:
                lines.append(f"      Examples for '{item['keyword']}':")
                for idx, headline_text in enumerate(headlines):
                    lines.append(f"        {idx+1}. {headline_text}")

    if len(keyword_data_list) > top_n_details:
        lines.append(f"    ... and {len(keyword_data_list) - top_n_details} more.")
    return lines


# --- Main Orchestration ---
def main():
    execution_date = datetime.date.today()
    print(
        f"Trend Analysis Report for data up to: {execution_date.strftime('%Y-%m-%d')}\n"
    )

    # Initialize report string variables
    daily_trends_report_str = ""
    weekly_trends_report_str = ""
    monthly_trends_report_str = ""

    # Variable to store the latest date from daily trends
    latest_report_date = None

    date_ranges_for_windows = get_date_ranges(execution_date)
    all_window_counts = {}

    # 1. Fetch and Aggregate data for each time window
    print("--- Period Summaries ---")
    for window_name, dates in date_ranges_for_windows.items():
        print(
            f"\nProcessing: {window_name.upper()} (From {dates['start']} to {dates['end']})"
        )
        items_for_window = get_items_for_date_range(dates["start"], dates["end"])

        # Special handling for daily window - fallback to last 2 days if no data
        if window_name == "daily" and not items_for_window:
            print(f"No items found for today and yesterday. Checking last 2 days...")
            fallback_ranges = get_date_ranges_with_fallback(
                execution_date, "daily", fallback_days=2
            )
            fallback_dates = fallback_ranges["daily"]
            print(
                f"Fallback: Checking from {fallback_dates['start']} to {fallback_dates['end']}"
            )
            items_for_window = get_items_for_date_range(
                fallback_dates["start"], fallback_dates["end"]
            )
            if items_for_window:
                # Update the date ranges for this window
                date_ranges_for_windows["daily"] = fallback_dates

        # Get latest date from daily window items
        if window_name == "daily" and items_for_window:
            latest_report_date = get_latest_date_from_items(items_for_window)

        if not items_for_window:
            print(f"No items found for {window_name} window.")
            all_window_counts[window_name] = defaultdict(Counter)
            continue

        aggregated_counts = aggregate_keyword_counts(items_for_window)
        all_window_counts[window_name] = aggregated_counts

        top_n = identify_top_n_keywords(aggregated_counts, n=TOP_N_FOR_PERIOD_SUMMARY)
        print(f"Top {TOP_N_FOR_PERIOD_SUMMARY} keywords for {window_name}:")
        has_any_top_keywords = False
        for field, top_list in top_n.items():
            if top_list:
                has_any_top_keywords = True
                print(f"  {field.replace('_', ' ').title()}: {top_list}")
        if not has_any_top_keywords:
            print("  No prominent keywords found for any field in this period.")
    print("\n" + "=" * 60 + "\n")

    # 2. Perform and Print Emerging Trend Comparisons
    # For daily trends, we need to adjust the baseline period
    trend_comparison_configs = [
        {
            "name": "Daily Emerging Trends",
            "current_period": "daily",
            "baseline_period": "weekly",  # This will be adjusted to previous 2-day period
        },
        {
            "name": "Weekly Emerging Trends",
            "current_period": "weekly",
            "baseline_period": "monthly",
        },
        {
            "name": "Monthly Emerging Trends",
            "current_period": "monthly",
            "baseline_period": "quarterly",
        },
    ]

    for config in trend_comparison_configs:
        current_report_lines = []
        report_name = config["name"]
        current_report_lines.append(f"--- {report_name} ---")

        current_win = config["current_period"]
        baseline_win = config["baseline_period"]

        # For daily comparison, we need to get the previous 2-day period data
        if current_win == "daily":
            # Get the current daily period dates
            current_dates = date_ranges_for_windows["daily"]
            current_start = datetime.datetime.strptime(
                current_dates["start"], "%Y-%m-%d"
            ).date()

            # Calculate previous 2-day period
            prev_end = current_start - datetime.timedelta(days=1)
            prev_start = prev_end - datetime.timedelta(days=1)

            print(f"\nFetching previous 2-day period data for daily comparison...")
            print(
                f"Previous period: {prev_start.strftime('%Y-%m-%d')} to {prev_end.strftime('%Y-%m-%d')}"
            )

            prev_items = get_items_for_date_range(
                prev_start.strftime("%Y-%m-%d"), prev_end.strftime("%Y-%m-%d")
            )

            if prev_items:
                prev_aggregated_counts = aggregate_keyword_counts(prev_items)
                # Use the previous period as baseline for daily
                baseline_counts = prev_aggregated_counts
                days_baseline = 2  # Previous 2-day period
            else:
                print("No data found for previous 2-day period. Using weekly baseline.")
                baseline_counts = all_window_counts.get(
                    baseline_win, defaultdict(Counter)
                )
                days_baseline = TIME_WINDOWS[baseline_win]
        else:
            baseline_counts = all_window_counts.get(baseline_win, defaultdict(Counter))
            days_baseline = TIME_WINDOWS[baseline_win]

        if (
            current_win not in all_window_counts
            or not all_window_counts[current_win]
            or not baseline_counts
        ):
            print(
                f"Insufficient data for {current_win} or baseline period. Skipping {report_name}.\n"
            )
            current_report_lines.append(
                f"\nInsufficient data for {current_win} or baseline period. Report generation skipped."
            )
            current_report_lines.append("\n" + "=" * 60 + "\n")
            if report_name == "Daily Emerging Trends":
                daily_trends_report_str = "\n".join(current_report_lines)
            elif report_name == "Weekly Emerging Trends":
                weekly_trends_report_str = "\n".join(current_report_lines)
            elif report_name == "Monthly Emerging Trends":
                monthly_trends_report_str = "\n".join(current_report_lines)
            continue

        current_counts = all_window_counts[current_win]
        days_current = TIME_WINDOWS[current_win]

        # For daily comparison with previous 2-day period, skip the baseline adjustment
        if current_win == "daily" and days_baseline == 2:
            # Direct comparison with previous period
            adjusted_baseline = baseline_counts
        else:
            if days_baseline <= days_current:
                print(
                    f"Baseline window must be longer than current window. Skipping {report_name}.\n"
                )
                current_report_lines.append(
                    f"\nBaseline window must be longer than current window. Report generation skipped."
                )
                current_report_lines.append("\n" + "=" * 60 + "\n")
                if report_name == "Daily Emerging Trends":
                    daily_trends_report_str = "\n".join(current_report_lines)
                elif report_name == "Weekly Emerging Trends":
                    weekly_trends_report_str = "\n".join(current_report_lines)
                elif report_name == "Monthly Emerging Trends":
                    monthly_trends_report_str = "\n".join(current_report_lines)
                continue

            adjusted_baseline = calculate_adjusted_baseline_counts(
                current_counts, baseline_counts, days_current, days_baseline
            )

        period_trends = compare_trends(
            current_counts, adjusted_baseline, SIGNIFICANT_INCREASE_THRESHOLD
        )

        all_newly_emerging = []
        for field, kws_data in period_trends["newly_emerging"].items():
            for kw, count_val in kws_data:  # Renamed count to count_val
                all_newly_emerging.append(
                    {"keyword": kw, "field": field, "count": count_val}
                )
        all_newly_emerging.sort(key=lambda x: x["count"], reverse=True)

        current_report_lines.extend(
            format_trend_report_section(
                title="Newly Appearing Keywords (vs. avg. of prior periods from baseline)",
                keyword_data_list=all_newly_emerging,
                item_type="new",
                date_ranges_for_windows=date_ranges_for_windows,
                current_win_name=current_win,
                container_client=container,  # Pass the global container client
                debug_print_mode=DEBUG_PRINT,
                top_n_details=TOP_N_FOR_EMERGING_TREND_DETAILS,
                fetch_headlines_top_n=3,
            )
        )

        all_significantly_increased = []
        for field, kws_data in period_trends["significantly_increased"].items():
            for kw, cur_count, prev_avg, factor in kws_data:
                all_significantly_increased.append(
                    {
                        "keyword": kw,
                        "field": field,
                        "current_count": cur_count,
                        "prev_avg": prev_avg,
                        "factor": factor,
                    }
                )
        all_significantly_increased.sort(
            key=lambda x: (x["factor"], x["current_count"]), reverse=True
        )

        current_report_lines.extend(
            format_trend_report_section(
                title="Rapidly Growing Keywords",
                keyword_data_list=all_significantly_increased,
                item_type="growing",
                date_ranges_for_windows=date_ranges_for_windows,
                current_win_name=current_win,
                container_client=container,  # Pass the global container client
                significant_increase_threshold_for_title=SIGNIFICANT_INCREASE_THRESHOLD,
                debug_print_mode=DEBUG_PRINT,
                top_n_details=TOP_N_FOR_EMERGING_TREND_DETAILS,
                fetch_headlines_top_n=3,
            )
        )

        current_report_lines.append("\n" + "=" * 60 + "\n")

        if report_name == "Daily Emerging Trends":
            daily_trends_report_str = "\n".join(current_report_lines)
        elif report_name == "Weekly Emerging Trends":
            weekly_trends_report_str = "\n".join(current_report_lines)
        elif report_name == "Monthly Emerging Trends":
            monthly_trends_report_str = "\n".join(current_report_lines)

    print(
        "Trend analysis complete."
    )  # This is the clear ending of the script's active output

    # Save reports to knowledge-reports container
    if latest_report_date:
        print(
            f"\nSaving reports to knowledge-reports container with date: {latest_report_date}"
        )

        if daily_trends_report_str:
            save_trend_report(daily_trends_report_str, "daily", latest_report_date)

        if weekly_trends_report_str:
            save_trend_report(weekly_trends_report_str, "weekly", latest_report_date)

        if monthly_trends_report_str:
            save_trend_report(monthly_trends_report_str, "monthly", latest_report_date)
    else:
        print(
            "\nWarning: Could not determine latest report date from daily trends. Reports not saved."
        )

    # --- You can uncomment the lines below to print the generated report strings ---
    # print("\n\n--- Generated Report Strings (for verification) ---")
    print("\n--- Daily Emerging Trends Report String ---")
    print(
        daily_trends_report_str
        if daily_trends_report_str
        else "Report not generated or empty."
    )
    print("\n--- Weekly Emerging Trends Report String ---")
    print(
        weekly_trends_report_str
        if weekly_trends_report_str
        else "Report not generated or empty."
    )
    print("\n--- Monthly Emerging Trends Report String ---")
    print(
        monthly_trends_report_str
        if monthly_trends_report_str
        else "Report not generated or empty."
    )
    # print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
