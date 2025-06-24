# Date and Time Utilities with Boost

*Duration: 1 week*

## Overview

Date and time handling is one of the most challenging aspects of programming due to complexities like time zones, daylight saving time, leap years, and varying calendar systems. Boost provides powerful libraries to handle these complexities efficiently and correctly.

This section covers Boost's comprehensive date and time handling libraries, including calendar operations, time zone support, high-resolution timing capabilities, and business logic implementations.

### Why Date/Time Programming is Complex

**Common Challenges:**
- **Time Zones**: Different regions have different UTC offsets
- **Daylight Saving Time**: Rules vary by location and change over time
- **Leap Years**: February 29th exists only in certain years
- **Leap Seconds**: Occasionally added to UTC to account for Earth's rotation
- **Calendar Systems**: Gregorian, Julian, and other calendar systems
- **Precision**: Different applications need different time precisions
- **Arithmetic**: Adding months/years can be ambiguous

**Real-world Problems Boost Solves:**
```cpp
// WRONG: Naive approach that fails
std::time_t now = std::time(nullptr);
now += 30 * 24 * 60 * 60;  // Add "30 days" - but months vary!

// RIGHT: Boost approach
bg::date today = bg::day_clock::local_day();
bg::date next_month = today + bg::months(1);  // Handles month boundaries correctly
```

## Learning Topics

### Boost.DateTime - The Foundation
Understanding date and time representation, manipulation, and complex calendar calculations.

### Boost.Chrono - High Precision Timing  
For performance measurements, benchmarking, and high-resolution timing requirements.

### Time Zone Management
Handling global applications with proper time zone conversions and DST handling.

### Business Logic Implementation
Real-world applications like business day calculations, financial settlements, and scheduling.

## Fundamental Concepts

### Date and Time Representation

**Core Components Hierarchy:**
```
Date (Year, Month, Day)
    ↓
Time Duration (Hours, Minutes, Seconds, Fractional Seconds)
    ↓
DateTime (Date + Time Duration)
    ↓
Local DateTime (DateTime + Time Zone)
```

**Boost.DateTime Architecture:**
```cpp
// Boost Date/Time Type Hierarchy
namespace boost {
    namespace gregorian {
        class date;           // Represents a calendar date
        class date_duration;  // Represents a span of days
        class date_period;    // Represents a date range
    }
    
    namespace posix_time {
        class time_duration;  // Represents time spans
        class ptime;         // Represents point in time (date + time)
        class time_period;   // Represents time ranges
    }
    
    namespace local_time {
        class local_date_time; // Represents local time with time zone
        class time_zone;       // Represents time zone rules
    }
}
```

### Key Design Principles

**1. Immutability**
Date and time objects are immutable - operations return new objects rather than modifying existing ones.

```cpp
bg::date original(2023, 6, 21);
bg::date modified = original + bg::days(7);  // original unchanged
std::cout << "Original: " << original << "\n";   // 2023-Jun-21
std::cout << "Modified: " << modified << "\n";   // 2023-Jun-28
```

**2. Type Safety**
Different types prevent common errors like adding dates to dates.

```cpp
bg::date date1(2023, 6, 21);
bg::date date2(2023, 6, 28);

// COMPILE ERROR: Can't add dates
// bg::date result = date1 + date2;  // Won't compile!

// CORRECT: Subtract dates to get duration
bg::date_duration diff = date2 - date1;  // 7 days
```

**3. Precision Control**
Different types provide different levels of precision as needed.

```cpp
// Different precision levels
bg::date just_date(2023, 6, 21);                    // Day precision
bp::ptime with_time(just_date, bp::hours(14));      // Hour precision  
bp::ptime precise(just_date, bp::time_duration(14, 30, 15, 500)); // Sub-second precision
```

## Code Examples

### Basic Date Operations

#### Understanding Date Construction and Properties

```cpp
#include <boost/date_time/gregorian/gregorian.hpp>
#include <iostream>
#include <stdexcept>

namespace bg = boost::gregorian;

void demonstrate_date_construction() {
    std::cout << "=== Date Construction Methods ===\n";
    
    // Method 1: Direct construction (most common)
    bg::date explicit_date(2023, 6, 21);
    std::cout << "Explicit construction: " << explicit_date << "\n";
    
    // Method 2: Current date
    bg::date today = bg::day_clock::local_day();
    bg::date utc_today = bg::day_clock::universal_day();
    std::cout << "Local today: " << today << "\n";
    std::cout << "UTC today: " << utc_today << "\n";
    
    // Method 3: From string (various formats)
    try {
        bg::date from_iso = bg::from_string("2023-06-21");      // ISO format
        bg::date from_us = bg::from_us_string("06/21/2023");    // US format
        bg::date from_uk = bg::from_uk_string("21/06/2023");    // UK format
        
        std::cout << "From ISO string: " << from_iso << "\n";
        std::cout << "From US string: " << from_us << "\n";
        std::cout << "From UK string: " << from_uk << "\n";
    } catch (const std::exception& e) {
        std::cout << "Parse error: " << e.what() << "\n";
    }
    
    // Method 4: From day number (advanced)
    bg::date epoch(1970, 1, 1);
    long days_since_epoch = 19532;  // Example: days since 1970-01-01
    bg::date from_days = epoch + bg::days(days_since_epoch);
    std::cout << "From day number: " << from_days << "\n";
    
    // Method 5: Special dates
    bg::date not_a_date = bg::date(bg::not_a_date_time);
    bg::date pos_infinity = bg::date(bg::pos_infin);
    bg::date neg_infinity = bg::date(bg::neg_infin);
    
    std::cout << "Special dates:\n";
    std::cout << "  Not-a-date: " << not_a_date << " (is_not_a_date: " 
              << not_a_date.is_not_a_date() << ")\n";
    std::cout << "  Positive infinity: " << pos_infinity << " (is_infinity: " 
              << pos_infinity.is_infinity() << ")\n";
}

void demonstrate_date_properties() {
    std::cout << "\n=== Date Properties and Information ===\n";
    
    bg::date summer_solstice(2023, 6, 21);
    
    // Basic properties
    std::cout << "Date: " << summer_solstice << "\n";
    std::cout << "Year: " << summer_solstice.year() << "\n";
    std::cout << "Month: " << summer_solstice.month() << " (" 
              << summer_solstice.month().as_long_string() << ")\n";
    std::cout << "Day: " << summer_solstice.day() << "\n";
    
    // Advanced properties
    std::cout << "Day of week: " << summer_solstice.day_of_week() 
              << " (" << summer_solstice.day_of_week().as_long_string() << ")\n";
    std::cout << "Day of year: " << summer_solstice.day_of_year() << "\n";
    std::cout << "Week number: " << summer_solstice.week_number() << "\n";
    
    // Year properties
    bg::greg_year year = summer_solstice.year();
    std::cout << "Is leap year: " << std::boolalpha 
              << bg::gregorian_calendar::is_leap_year(year) << "\n";
    std::cout << "Days in year: " 
              << bg::gregorian_calendar::end_of_year_day(year) << "\n";
    
    // Month properties
    bg::greg_month month = summer_solstice.month();
    std::cout << "Days in month: " 
              << bg::gregorian_calendar::end_of_month_day(year, month) << "\n";
    
    // Calculate end of month and year
    bg::date end_of_month = summer_solstice.end_of_month();
    bg::date end_of_year = bg::date(year, 12, 31);
    
    std::cout << "End of this month: " << end_of_month << "\n";
    std::cout << "End of this year: " << end_of_year << "\n";
}

#### Date Arithmetic and Comparisons

Date arithmetic in Boost is intuitive but handles edge cases correctly that naive approaches miss.

```cpp
void demonstrate_date_arithmetic() {
    std::cout << "\n=== Date Arithmetic ===\n";
    
    bg::date base_date(2023, 1, 31);  // January 31st
    std::cout << "Base date: " << base_date << "\n";
    
    // Adding different duration types
    bg::date plus_days = base_date + bg::days(7);
    bg::date plus_weeks = base_date + bg::weeks(2);
    bg::date plus_months = base_date + bg::months(1);  // Smart month handling!
    bg::date plus_years = base_date + bg::years(1);
    
    std::cout << "Plus 7 days: " << plus_days << "\n";        // Feb 7
    std::cout << "Plus 2 weeks: " << plus_weeks << "\n";      // Feb 14  
    std::cout << "Plus 1 month: " << plus_months << "\n";     // Feb 28 (not Mar 3!)
    std::cout << "Plus 1 year: " << plus_years << "\n";       // Jan 31, 2024
    
    // Demonstrate month-end handling
    std::cout << "\n--- Month-end Arithmetic ---\n";
    std::vector<bg::date> month_ends = {
        bg::date(2023, 1, 31),  // Jan 31
        bg::date(2023, 3, 31),  // Mar 31  
        bg::date(2023, 5, 31),  // May 31
        bg::date(2023, 12, 31)  // Dec 31
    };
    
    for (const auto& date : month_ends) {
        bg::date next_month = date + bg::months(1);
        std::cout << date << " + 1 month = " << next_month << "\n";
    }
    
    // Leap year handling
    std::cout << "\n--- Leap Year Handling ---\n";
    bg::date leap_day(2024, 2, 29);  // 2024 is a leap year
    bg::date next_year = leap_day + bg::years(1);
    std::cout << "Leap day + 1 year: " << leap_day << " -> " << next_year << "\n";
    
    // Negative arithmetic
    std::cout << "\n--- Negative Arithmetic ---\n";
    bg::date recent = bg::day_clock::local_day();
    bg::date last_week = recent - bg::weeks(1);
    bg::date last_month = recent - bg::months(1);
    bg::date last_year = recent - bg::years(1);
    
    std::cout << "Today: " << recent << "\n";
    std::cout << "Last week: " << last_week << "\n";
    std::cout << "Last month: " << last_month << "\n";
    std::cout << "Last year: " << last_year << "\n";
}

void demonstrate_date_comparisons() {
    std::cout << "\n=== Date Comparisons and Sorting ===\n";
    
    std::vector<bg::date> dates = {
        bg::date(2023, 12, 25),  // Christmas
        bg::date(2023, 1, 1),    // New Year
        bg::date(2023, 7, 4),    // Independence Day
        bg::date(2023, 11, 23),  // Thanksgiving (example)
        bg::day_clock::local_day() // Today
    };
    
    std::cout << "Unsorted dates:\n";
    for (const auto& date : dates) {
        std::cout << "  " << date << "\n";
    }
    
    // Sort dates (dates are naturally comparable)
    std::sort(dates.begin(), dates.end());
    
    std::cout << "\nSorted dates:\n";
    for (const auto& date : dates) {
        std::cout << "  " << date << "\n";
    }
    
    // Compare with today
    bg::date today = bg::day_clock::local_day();
    std::cout << "\nComparison with today (" << today << "):\n";
    
    for (const auto& date : dates) {
        if (date < today) {
            bg::date_duration diff = today - date;
            std::cout << "  " << date << " was " << diff.days() << " days ago\n";
        } else if (date > today) {
            bg::date_duration diff = date - today;
            std::cout << "  " << date << " is " << diff.days() << " days away\n";
        } else {
            std::cout << "  " << date << " is today!\n";
        }
    }
    
    // Find min/max dates
    auto min_date = *std::min_element(dates.begin(), dates.end());
    auto max_date = *std::max_element(dates.begin(), dates.end());
    bg::date_duration span = max_date - min_date;
    
    std::cout << "\nDate range: " << min_date << " to " << max_date 
              << " (span: " << span.days() << " days)\n";
}
```

### Date Ranges, Periods, and Iterations

Understanding date ranges and how to work with collections of dates efficiently.

```cpp
void demonstrate_date_periods() {
    std::cout << "\n=== Date Periods and Ranges ===\n";
    
    // Create date periods
    bg::date start(2023, 6, 1);
    bg::date end(2023, 6, 30);
    bg::date_period june(start, end);
    
    std::cout << "June period: " << june << "\n";
    std::cout << "Period length: " << june.length().days() << " days\n";
    std::cout << "Period begin: " << june.begin() << "\n";
    std::cout << "Period end: " << june.end() << "\n";
    std::cout << "Period last: " << june.last() << "\n";
    
    // Test containment
    std::vector<bg::date> test_dates = {
        bg::date(2023, 5, 31),   // Before period
        bg::date(2023, 6, 1),    // First day
        bg::date(2023, 6, 15),   // Middle
        bg::date(2023, 6, 30),   // Last day  
        bg::date(2023, 7, 1)     // After period
    };
    
    std::cout << "\nContainment tests:\n";
    for (const auto& date : test_dates) {
        bool contains = june.contains(date);
        std::cout << "  " << date << ": " << (contains ? "IN" : "OUT") << "\n";
    }
    
    // Period operations
    bg::date_period q1(bg::date(2023, 1, 1), bg::date(2023, 3, 31));
    bg::date_period q2(bg::date(2023, 4, 1), bg::date(2023, 6, 30));
    bg::date_period first_half(bg::date(2023, 1, 1), bg::date(2023, 6, 30));
    
    std::cout << "\n--- Period Operations ---\n";
    std::cout << "Q1: " << q1 << "\n";
    std::cout << "Q2: " << q2 << "\n";
    std::cout << "First half: " << first_half << "\n";
    
    // Check adjacency
    bool adjacent = q1.is_adjacent(q2);
    std::cout << "Q1 and Q2 are adjacent: " << std::boolalpha << adjacent << "\n";
    
    // Check intersection
    bg::date_period intersection = q1.intersection(first_half);
    if (!intersection.is_null()) {
        std::cout << "Q1 ∩ First-half: " << intersection << "\n";
    }
    
    // Merge periods
    bg::date_period merged = q1.merge(q2);
    if (!merged.is_null()) {
        std::cout << "Q1 ∪ Q2: " << merged << "\n";
    }
    
    // Shift periods
    bg::date_period shifted = june.shift(bg::months(1));
    std::cout << "June shifted by 1 month: " << shifted << "\n";
}

void demonstrate_date_iteration() {
    std::cout << "\n=== Date Iteration Patterns ===\n";
    
    bg::date start(2023, 6, 1);
    bg::date end(2023, 6, 30);
    
    // Basic day iteration
    std::cout << "First week of June 2023:\n";
    bg::date current = start;
    for (int i = 0; i < 7 && current <= end; ++i, ++current) {
        std::cout << "  " << current << " (" << current.day_of_week() << ")\n";
    }
    
    // Using day_iterator (more elegant)
    std::cout << "\nWeekdays in first two weeks:\n";
    int weekday_count = 0;
    for (bg::day_iterator it(start); it <= end && weekday_count < 10; ++it) {
        if (it->day_of_week() != bg::Sunday && it->day_of_week() != bg::Saturday) {
            std::cout << "  " << *it << " (" << it->day_of_week() << ")\n";
            weekday_count++;
        }
    }
    
    // Week iteration
    std::cout << "\nMondays in June 2023:\n";
    bg::date june_start(2023, 6, 1);
    bg::date june_end(2023, 6, 30);
    
    // Find first Monday
    bg::date first_monday = june_start;
    while (first_monday.day_of_week() != bg::Monday && first_monday <= june_end) {
        first_monday += bg::days(1);
    }
    
    for (bg::week_iterator wit(first_monday); wit <= june_end; ++wit) {
        std::cout << "  " << *wit << "\n";
    }
    
    // Month iteration
    std::cout << "\nFirst day of each month in 2023:\n";
    bg::date year_start(2023, 1, 1);
    bg::date year_end(2023, 12, 31);
    
    for (bg::month_iterator mit(year_start); mit <= year_end; ++mit) {
        std::cout << "  " << *mit << " (" << mit->month().as_long_string() << ")\n";
    }
    
    // Year iteration (for multi-year spans)
    std::cout << "\nJanuary 1st for 5 years:\n";
    bg::date start_year(2023, 1, 1);
    for (bg::year_iterator yit(start_year); yit <= bg::date(2027, 12, 31); ++yit) {
        std::cout << "  " << *yit << "\n";
    }
}

### Time Duration and Time Point Operations

Understanding how to work with time durations and precise time points.

```cpp
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <iomanip>

namespace bp = boost::posix_time;
namespace bg = boost::gregorian;

void demonstrate_time_duration_basics() {
    std::cout << "\n=== Time Duration Fundamentals ===\n";
    
    // Various ways to create time durations
    bp::time_duration morning(8, 30, 0);                    // 8:30:00
    bp::time_duration afternoon = bp::hours(14) + bp::minutes(45) + bp::seconds(30);
    bp::time_duration precise = bp::time_duration(16, 20, 30, 750000); // With microseconds
    
    std::cout << "Morning: " << morning << "\n";
    std::cout << "Afternoon: " << afternoon << "\n";
    std::cout << "Precise: " << precise << "\n";
    
    // Duration properties
    std::cout << "\n--- Duration Properties ---\n";
    std::cout << "Afternoon duration breakdown:\n";
    std::cout << "  Hours: " << afternoon.hours() << "\n";
    std::cout << "  Minutes: " << afternoon.minutes() << "\n";
    std::cout << "  Seconds: " << afternoon.seconds() << "\n";
    std::cout << "  Total seconds: " << afternoon.total_seconds() << "\n";
    std::cout << "  Total milliseconds: " << afternoon.total_milliseconds() << "\n";
    std::cout << "  Total microseconds: " << afternoon.total_microseconds() << "\n";
    
    // Special durations
    bp::time_duration not_a_time = bp::time_duration(bp::not_a_date_time);
    bp::time_duration pos_infinity = bp::time_duration(bp::pos_infin);
    bp::time_duration neg_infinity = bp::time_duration(bp::neg_infin);
    
    std::cout << "\n--- Special Durations ---\n";
    std::cout << "Not-a-time: " << not_a_time << " (is_not_a_date_time: " 
              << not_a_time.is_not_a_date_time() << ")\n";
    std::cout << "Positive infinity: " << pos_infinity << "\n";
    std::cout << "Negative infinity: " << neg_infinity << "\n";
}

void demonstrate_time_arithmetic() {
    std::cout << "\n=== Time Duration Arithmetic ===\n";
    
    bp::time_duration meeting_start(9, 0, 0);      // 9:00 AM
    bp::time_duration meeting_length(1, 30, 0);    // 1.5 hours
    bp::time_duration break_time(0, 15, 0);        // 15 minutes
    
    std::cout << "Meeting start: " << meeting_start << "\n";
    std::cout << "Meeting length: " << meeting_length << "\n";
    std::cout << "Break time: " << break_time << "\n";
    
    // Addition and subtraction
    bp::time_duration morning_end = meeting_start + meeting_length;
    bp::time_duration afternoon_start = morning_end + break_time;
    
    std::cout << "Morning session ends: " << morning_end << "\n";
    std::cout << "Afternoon session starts: " << afternoon_start << "\n";
    
    // Time remaining calculations
    bp::time_duration current_time(10, 45, 0);  // 10:45 AM
    
    if (current_time > meeting_start && current_time < morning_end) {
        bp::time_duration remaining = morning_end - current_time;
        std::cout << "Meeting time remaining: " << remaining << "\n";
    }
    
    // Multiplication and division
    bp::time_duration daily_commute(0, 45, 0);  // 45 minutes
    bp::time_duration weekly_commute = daily_commute * 10;  // 5 days * 2 trips
    bp::time_duration average_trip = weekly_commute / 10;
    
    std::cout << "Daily commute (round trip): " << daily_commute << "\n";
    std::cout << "Weekly commute: " << weekly_commute << "\n";
    std::cout << "Average trip: " << average_trip << "\n";
    
    // Comparison operations
    std::vector<bp::time_duration> durations = {
        bp::minutes(30), bp::hours(1), bp::seconds(90), bp::minutes(90)
    };
    
    std::cout << "\n--- Duration Comparisons ---\n";
    for (size_t i = 0; i < durations.size(); ++i) {
        for (size_t j = i + 1; j < durations.size(); ++j) {
            std::cout << durations[i] << " vs " << durations[j] << ": ";
            if (durations[i] < durations[j]) {
                std::cout << "< \n";
            } else if (durations[i] > durations[j]) {
                std::cout << "> \n";
            } else {
                std::cout << "= \n";
            }
        }
    }
}

void demonstrate_datetime_operations() {
    std::cout << "\n=== DateTime Operations ===\n";
    
    // Create date-time points
    bg::date today = bg::day_clock::local_day();
    bp::time_duration now_time(14, 30, 0);  // 2:30 PM
    bp::ptime now(today, now_time);
    
    std::cout << "Current date-time: " << now << "\n";
    
    // Alternative construction methods
    bp::ptime from_string = bp::time_from_string("2023-06-21 16:45:30");
    bp::ptime current_time = bp::second_clock::local_time();
    bp::ptime precise_time = bp::microsec_clock::local_time();
    
    std::cout << "From string: " << from_string << "\n";
    std::cout << "Current (second precision): " << current_time << "\n";
    std::cout << "Current (microsecond precision): " << precise_time << "\n";
    
    // Extract components
    std::cout << "\n--- DateTime Components ---\n";
    std::cout << "Date part: " << now.date() << "\n";
    std::cout << "Time part: " << now.time_of_day() << "\n";
    
    // DateTime arithmetic with mixed types
    bp::ptime meeting_time(bg::date(2023, 6, 21), bp::time_duration(14, 0, 0));
    
    // Add different duration types
    bp::ptime delayed_meeting = meeting_time + bp::minutes(15);  // Time duration
    bp::ptime next_week_meeting = meeting_time + bg::days(7);   // Date duration
    bp::ptime monthly_meeting = meeting_time + bg::months(1);   // Month duration
    
    std::cout << "\n--- Meeting Schedule ---\n";
    std::cout << "Original meeting: " << meeting_time << "\n";
    std::cout << "Delayed 15 min: " << delayed_meeting << "\n";
    std::cout << "Next week: " << next_week_meeting << "\n";
    std::cout << "Next month: " << monthly_meeting << "\n";
    
    // Calculate durations between time points
    bp::time_duration time_until_meeting = meeting_time - now;
    if (!time_until_meeting.is_negative()) {
        std::cout << "Time until meeting: " << time_until_meeting << "\n";
        
        long hours_until = time_until_meeting.hours();
        long minutes_until = time_until_meeting.minutes() % 60;
        std::cout << "In other words: " << hours_until << " hours and " 
                  << minutes_until << " minutes\n";
    } else {
        bp::time_duration time_since = -time_until_meeting;
        std::cout << "Meeting was " << time_since << " ago\n";
    }
    
    // Time periods with ptime
    bp::time_period work_day(
        bp::ptime(today, bp::time_duration(9, 0, 0)),   // 9 AM
        bp::ptime(today, bp::time_duration(17, 0, 0))   // 5 PM
    );
    
    std::cout << "\n--- Work Day Period ---\n";
    std::cout << "Work day: " << work_day << "\n";
    std::cout << "Length: " << work_day.length() << "\n";
    
    // Check if current time is within work hours
    if (work_day.contains(now)) {
        std::cout << "Currently within work hours\n";
        bp::time_duration time_left = work_day.end() - now;
        std::cout << "Time left in work day: " << time_left << "\n";
    } else {
        std::cout << "Currently outside work hours\n";
    }
}

### Advanced Parsing and Formatting

Boost provides powerful and flexible parsing and formatting capabilities for different date/time formats and locales.

```cpp
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <locale>
#include <sstream>
#include <vector>

namespace bg = boost::gregorian;
namespace bp = boost::posix_time;

void demonstrate_comprehensive_parsing() {
    std::cout << "\n=== Comprehensive Date/Time Parsing ===\n";
    
    // Date parsing from various formats
    struct DateParseTest {
        std::string input;
        std::string description;
        std::function<bg::date()> parser;
    };
    
    std::vector<DateParseTest> date_tests = {
        {"2023-06-21", "ISO format", []() { return bg::from_string("2023-06-21"); }},
        {"2023/06/21", "Slash format", []() { return bg::from_us_string("06/21/2023"); }},
        {"21/06/2023", "UK format", []() { return bg::from_uk_string("21/06/2023"); }},
        {"Jun 21, 2023", "Named month", []() { return bg::from_string("2023-Jun-21"); }},
        {"20230621", "Compact format", []() { return bg::from_undelimited_string("20230621"); }}
    };
    
    std::cout << "Date parsing tests:\n";
    for (const auto& test : date_tests) {
        try {
            bg::date parsed = test.parser();
            std::cout << "  ✓ " << std::setw(15) << test.input 
                      << " (" << test.description << ") -> " << parsed << "\n";
        } catch (const std::exception& e) {
            std::cout << "  ✗ " << std::setw(15) << test.input 
                      << " (" << test.description << ") -> ERROR: " << e.what() << "\n";
        }
    }
    
    // DateTime parsing
    std::cout << "\n--- DateTime Parsing ---\n";
    std::vector<std::string> datetime_strings = {
        "2023-06-21 14:30:15",
        "2023-06-21T14:30:15",  // ISO 8601
        "2023-06-21 14:30:15.500",  // With fractional seconds
        "20230621T143015"  // Compact ISO
    };
    
    for (const auto& dt_str : datetime_strings) {
        try {
            bp::ptime parsed;
            
            if (dt_str.find('T') != std::string::npos) {
                // ISO format with T separator
                if (dt_str.length() == 15 && dt_str.find('.') == std::string::npos && 
                    dt_str.find(':') == std::string::npos) {
                    // Compact ISO format
                    parsed = bp::from_iso_string(dt_str);
                } else {
                    // Standard ISO format
                    parsed = bp::from_iso_extended_string(dt_str);
                }
            } else {
                // Space-separated format
                parsed = bp::time_from_string(dt_str);
            }
            
            std::cout << "  ✓ " << std::setw(25) << dt_str << " -> " << parsed << "\n";
        } catch (const std::exception& e) {
            std::cout << "  ✗ " << std::setw(25) << dt_str << " -> ERROR: " << e.what() << "\n";
        }
    }
}

void demonstrate_custom_formatting() {
    std::cout << "\n=== Custom Formatting with Facets ===\n";
    
    bg::date date(2023, 6, 21);
    bp::ptime datetime(date, bp::time_duration(14, 30, 15));
    
    // Standard formatting options
    std::cout << "Standard formats:\n";
    std::cout << "  Simple: " << to_simple_string(date) << "\n";
    std::cout << "  ISO: " << to_iso_string(date) << "\n";
    std::cout << "  ISO Extended: " << to_iso_extended_string(date) << "\n";
    std::cout << "  DateTime ISO: " << to_iso_string(datetime) << "\n";
    std::cout << "  DateTime Extended: " << to_iso_extended_string(datetime) << "\n";
    
    // Custom date formatting using facets
    std::cout << "\n--- Custom Date Formats ---\n";
    
    struct FormatTest {
        std::string format;
        std::string description;
    };
    
    std::vector<FormatTest> formats = {
        {"%A, %B %d, %Y", "Full weekday and month"},
        {"%a %b %d, %Y", "Abbreviated day and month"},
        {"%m/%d/%Y", "US numeric format"},
        {"%d/%m/%Y", "European numeric format"},
        {"%Y年%m月%d日", "Chinese-style format"},
        {"%B %Y", "Month and year only"},
        {"%j", "Day of year"}
    };
    
    for (const auto& fmt : formats) {
        try {
            std::ostringstream ss;
            ss.imbue(std::locale(std::cout.getloc(), 
                               new bg::date_facet(fmt.format.c_str())));
            ss << date;
            std::cout << "  " << std::setw(20) << fmt.format 
                      << " (" << fmt.description << "): " << ss.str() << "\n";
        } catch (const std::exception& e) {
            std::cout << "  " << std::setw(20) << fmt.format 
                      << " -> ERROR: " << e.what() << "\n";
        }
    }
    
    // Custom time formatting
    std::cout << "\n--- Custom Time Formats ---\n";
    
    std::vector<FormatTest> time_formats = {
        {"%H:%M:%S", "24-hour format"},
        {"%I:%M:%S %p", "12-hour format with AM/PM"},
        {"%H時%M分%S秒", "Japanese-style time"},
        {"%H:%M", "Hours and minutes only"},
        {"%S", "Seconds only"},
        {"%f", "Fractional seconds"}
    };
    
    for (const auto& fmt : time_formats) {
        try {
            std::ostringstream ss;
            ss.imbue(std::locale(std::cout.getloc(), 
                               new bp::time_facet(fmt.format.c_str())));
            ss << datetime;
            std::cout << "  " << std::setw(20) << fmt.format 
                      << " (" << fmt.description << "): " << ss.str() << "\n";
        } catch (const std::exception& e) {
            std::cout << "  " << std::setw(20) << fmt.format 
                      << " -> ERROR: " << e.what() << "\n";
        }
    }
}

void demonstrate_input_parsing() {
    std::cout << "\n=== Input Stream Parsing ===\n";
    
    // Parse dates from input stream
    std::string date_input = "2023-06-21 2023/12/25 21-Dec-2023";
    std::istringstream date_stream(date_input);
    
    std::cout << "Parsing dates from stream: \"" << date_input << "\"\n";
    
    bg::date parsed_date;
    while (date_stream >> parsed_date) {
        std::cout << "  Parsed: " << parsed_date << "\n";
    }
    
    // Parse times from input stream with custom format
    std::string time_input = "14:30:15 09:45:30 23:59:59";
    std::istringstream time_stream(time_input);
    
    std::cout << "\nParsing times from stream: \"" << time_input << "\"\n";
    
    bp::time_duration parsed_time;
    while (time_stream >> parsed_time) {
        std::cout << "  Parsed: " << parsed_time << "\n";
    }
    
    // Advanced: Parse with custom input facet
    std::cout << "\n--- Custom Input Format ---\n";
    
    std::string custom_input = "21/06/2023 25/12/2023";  // DD/MM/YYYY format
    std::istringstream custom_stream(custom_input);
    
    // Set up input facet for DD/MM/YYYY format
    custom_stream.imbue(std::locale(custom_stream.getloc(), 
                                  new bg::date_input_facet("%d/%m/%Y")));
    
    std::cout << "Parsing DD/MM/YYYY format: \"" << custom_input << "\"\n";
    
    bg::date custom_parsed;
    while (custom_stream >> custom_parsed) {
        std::cout << "  Parsed: " << custom_parsed << "\n";
    }
}

void demonstrate_error_handling() {
    std::cout << "\n=== Error Handling in Parsing ===\n";
    
    std::vector<std::string> problematic_inputs = {
        "2023-02-30",  // Invalid date (Feb 30th)
        "2023-13-01",  // Invalid month
        "2023-12-32",  // Invalid day
        "not-a-date",  // Invalid format
        "2023/2/29",   // Feb 29 in non-leap year
        ""             // Empty string
    };
    
    for (const auto& input : problematic_inputs) {
        std::cout << "Testing: \"" << input << "\" -> ";
        
        try {
            if (input.empty()) {
                std::cout << "Empty input (skipped)\n";
                continue;
            }
            
            bg::date parsed = bg::from_string(input);
            std::cout << "✓ Parsed as: " << parsed << "\n";
            
        } catch (const bg::bad_month& e) {
            std::cout << "✗ Bad month: " << e.what() << "\n";
        } catch (const bg::bad_day_of_year& e) {
            std::cout << "✗ Bad day of year: " << e.what() << "\n";
        } catch (const bg::bad_day_of_month& e) {
            std::cout << "✗ Bad day of month: " << e.what() << "\n";
        } catch (const std::exception& e) {
            std::cout << "✗ Parse error: " << e.what() << "\n";
        }
    }
    
    // Safe parsing helper function
    auto safe_parse_date = [](const std::string& input) -> std::optional<bg::date> {
        try {
            return bg::from_string(input);
        } catch (const std::exception&) {
            return std::nullopt;
        }
    };
    
    std::cout << "\n--- Safe Parsing Results ---\n";
    for (const auto& input : problematic_inputs) {
        if (auto result = safe_parse_date(input)) {
            std::cout << "\"" << input << "\" -> " << *result << "\n";
        } else {
            std::cout << "\"" << input << "\" -> Failed to parse\n";
        }
    }
}

### Time Zones and Local Time
```cpp
#include <boost/date_time/local_time/local_time.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>

namespace bl = boost::local_time;
namespace bp = boost::posix_time;
namespace bg = boost::gregorian;

void demonstrate_time_zones() {
    // Create time zone objects
    bl::time_zone_ptr ny_tz(new bl::posix_time_zone("EST-5EDT,M4.1.0,M10.5.0"));
    bl::time_zone_ptr la_tz(new bl::posix_time_zone("PST-8PDT,M4.1.0,M10.5.0"));
    bl::time_zone_ptr london_tz(new bl::posix_time_zone("GMT0BST,M3.5.0/1,M10.5.0/2"));
    
    // Create local times
    bp::ptime utc_time(bg::date(2023, 6, 21), bp::time_duration(18, 30, 0));
    
    bl::local_date_time ny_time(utc_time, ny_tz);
    bl::local_date_time la_time(utc_time, la_tz);
    bl::local_date_time london_time(utc_time, london_tz);
    
    std::cout << "UTC time: " << utc_time << "\n";
    std::cout << "New York: " << ny_time << "\n";
    std::cout << "Los Angeles: " << la_time << "\n";
    std::cout << "London: " << london_time << "\n";
    
    // Convert between time zones
    bl::local_date_time ny_local(bg::date(2023, 6, 21), 
                                 bp::time_duration(14, 30, 0), ny_tz, false);
    
    std::cout << "\nMeeting at 2:30 PM in New York:\n";
    std::cout << "New York: " << ny_local << "\n";
    std::cout << "UTC: " << ny_local.utc_time() << "\n";
    
    // Convert to other time zones
    bl::local_date_time la_equivalent(ny_local.utc_time(), la_tz);
    bl::local_date_time london_equivalent(ny_local.utc_time(), london_tz);
    
    std::cout << "Los Angeles: " << la_equivalent << "\n";
    std::cout << "London: " << london_equivalent << "\n";
}
```

### High-Resolution Timing with Boost.Chrono
```cpp
#include <boost/chrono.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

namespace bc = boost::chrono;

void demonstrate_timing() {
    // Basic timing
    auto start = bc::high_resolution_clock::now();
    
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::vector<int> data(1000000);
    std::iota(data.begin(), data.end(), 1);
    std::sort(data.begin(), data.end(), std::greater<int>());
    
    auto end = bc::high_resolution_clock::now();
    auto duration = end - start;
    
    std::cout << "Operation took: " 
              << bc::duration_cast<bc::milliseconds>(duration).count() 
              << " milliseconds\n";
    
    std::cout << "In microseconds: "
              << bc::duration_cast<bc::microseconds>(duration).count() 
              << " μs\n";
}

class Timer {
public:
    void start() {
        start_time_ = bc::high_resolution_clock::now();
    }
    
    void stop() {
        end_time_ = bc::high_resolution_clock::now();
    }
    
    template<typename Duration>
    typename Duration::rep elapsed() const {
        return bc::duration_cast<Duration>(end_time_ - start_time_).count();
    }
    
    void print_elapsed() const {
        auto duration = end_time_ - start_time_;
        
        if (duration >= bc::seconds(1)) {
            std::cout << elapsed<bc::seconds>() << " seconds\n";
        } else if (duration >= bc::milliseconds(1)) {
            std::cout << elapsed<bc::milliseconds>() << " milliseconds\n";
        } else {
            std::cout << elapsed<bc::microseconds>() << " microseconds\n";
        }
    }
    
private:
    bc::high_resolution_clock::time_point start_time_;
    bc::high_resolution_clock::time_point end_time_;
};

void demonstrate_timer_class() {
    Timer timer;
    
    timer.start();
    
    // Simulate different workloads
    std::vector<double> results;
    for (int i = 0; i < 100000; ++i) {
        results.push_back(std::sin(i) * std::cos(i));
    }
    
    timer.stop();
    
    std::cout << "Mathematical operations: ";
    timer.print_elapsed();
}
```

### Business Date Calculations
```cpp
#include <boost/date_time/gregorian/gregorian.hpp>
#include <iostream>
#include <set>

namespace bg = boost::gregorian;

class BusinessCalendar {
public:
    BusinessCalendar() {
        // Add standard US holidays for 2023
        holidays_.insert(bg::date(2023, 1, 1));   // New Year's Day
        holidays_.insert(bg::date(2023, 1, 16));  // MLK Day
        holidays_.insert(bg::date(2023, 2, 20));  // Presidents Day
        holidays_.insert(bg::date(2023, 5, 29));  // Memorial Day
        holidays_.insert(bg::date(2023, 6, 19));  // Juneteenth
        holidays_.insert(bg::date(2023, 7, 4));   // Independence Day
        holidays_.insert(bg::date(2023, 9, 4));   // Labor Day
        holidays_.insert(bg::date(2023, 10, 9));  // Columbus Day
        holidays_.insert(bg::date(2023, 11, 11)); // Veterans Day
        holidays_.insert(bg::date(2023, 11, 23)); // Thanksgiving
        holidays_.insert(bg::date(2023, 12, 25)); // Christmas
    }
    
    bool isBusinessDay(const bg::date& date) const {
        // Check if it's a weekend
        if (date.day_of_week() == bg::Saturday || 
            date.day_of_week() == bg::Sunday) {
            return false;
        }
        
        // Check if it's a holiday
        return holidays_.find(date) == holidays_.end();
    }
    
    bg::date addBusinessDays(const bg::date& start, int business_days) const {
        bg::date current = start;
        int days_added = 0;
        
        while (days_added < business_days) {
            current += bg::days(1);
            if (isBusinessDay(current)) {
                days_added++;
            }
        }
        
        return current;
    }
    
    int countBusinessDays(const bg::date& start, const bg::date& end) const {
        int count = 0;
        bg::date current = start;
        
        while (current <= end) {
            if (isBusinessDay(current)) {
                count++;
            }
            current += bg::days(1);
        }
        
        return count;
    }
    
private:
    std::set<bg::date> holidays_;
};

void demonstrate_business_calendar() {
    BusinessCalendar calendar;
    
    bg::date start_date(2023, 6, 20);  // Tuesday
    std::cout << "Start date: " << start_date 
              << " (" << start_date.day_of_week() << ")\n";
    
    // Add 5 business days
    bg::date result = calendar.addBusinessDays(start_date, 5);
    std::cout << "5 business days later: " << result 
              << " (" << result.day_of_week() << ")\n";
    
    // Count business days in a period
    bg::date period_start(2023, 6, 1);
    bg::date period_end(2023, 6, 30);
    int business_days = calendar.countBusinessDays(period_start, period_end);
    
    std::cout << "Business days in June 2023: " << business_days << "\n";
    
    // Check specific dates
    std::vector<bg::date> test_dates = {
        bg::date(2023, 7, 4),   // Independence Day (holiday)
        bg::date(2023, 7, 3),   // Monday before
        bg::date(2023, 7, 8),   // Saturday (weekend)
        bg::date(2023, 7, 10)   // Monday after
    };
    
    std::cout << "\nBusiness day check:\n";
    for (const auto& date : test_dates) {
        std::cout << "  " << date << " (" << date.day_of_week() << "): "
                  << (calendar.isBusinessDay(date) ? "Business day" : "Non-business day") << "\n";
    }
}
```

## Practical Exercises

### Exercise 1: Event Scheduler System
**Objective**: Create a comprehensive meeting scheduler that handles time zones and finds optimal meeting times.

**Implementation Challenge**:
```cpp
class GlobalEventScheduler {
private:
    struct Meeting {
        std::string title;
        bl::local_date_time start_time;
        bp::time_duration duration;
        std::vector<std::string> attendees;
        std::string time_zone;
    };
    
    std::vector<Meeting> meetings_;
    std::map<std::string, bl::time_zone_ptr> time_zones_;
    
public:
    // TODO: Implement these methods
    void add_time_zone(const std::string& name, const std::string& posix_tz);
    bool schedule_meeting(const std::string& title, 
                         const std::string& tz_name,
                         const bp::ptime& utc_time,
                         const bp::time_duration& duration,
                         const std::vector<std::string>& attendees);
    
    std::vector<bp::time_period> find_free_slots(
        const std::string& tz_name,
        const bg::date& date,
        const bp::time_duration& min_duration,
        const bp::time_period& business_hours
    );
    
    bp::ptime find_optimal_meeting_time(
        const std::vector<std::string>& time_zones,
        const bg::date& preferred_date,
        const bp::time_duration& duration
    );
    
    void print_schedule_for_timezone(const std::string& tz_name, 
                                   const bg::date& date);
};

// Usage example:
int main() {
    GlobalEventScheduler scheduler;
    
    // Add time zones
    scheduler.add_time_zone("NYC", "EST-5EDT,M4.1.0,M10.5.0");
    scheduler.add_time_zone("London", "GMT0BST,M3.5.0/1,M10.5.0/2");
    scheduler.add_time_zone("Tokyo", "JST-9");
    
    // Schedule meetings and find optimal times
    std::vector<std::string> global_team = {"NYC", "London", "Tokyo"};
    auto optimal_time = scheduler.find_optimal_meeting_time(
        global_team, bg::date(2023, 6, 21), bp::hours(1)
    );
    
    std::cout << "Optimal meeting time (UTC): " << optimal_time << "\n";
    
    return 0;
}
```

**Learning Goals**:
- Master time zone conversions and calculations
- Handle daylight saving time transitions
- Implement complex scheduling algorithms
- Work with multiple time zones simultaneously

### Exercise 2: Financial Settlement Calculator
**Objective**: Implement business day calculations for financial settlements with holiday support.

**Implementation Challenge**:
```cpp
class FinancialCalendar {
public:
    enum class HolidayCalendar {
        US_FEDERAL,
        UK_BANK,
        JAPAN_NATIONAL,
        CUSTOM
    };
    
private:
    std::map<HolidayCalendar, std::set<bg::date>> holiday_calendars_;
    std::map<std::string, std::function<double(int)>> interest_curves_;
    
public:
    // TODO: Implement these methods
    void load_holiday_calendar(HolidayCalendar cal, const std::vector<bg::date>& holidays);
    
    bool is_business_day(const bg::date& date, HolidayCalendar cal);
    
    bg::date add_business_days(const bg::date& start, int business_days, 
                              HolidayCalendar cal);
    
    bg::date get_settlement_date(const bg::date& trade_date, int settlement_days,
                                HolidayCalendar cal);
    
    double calculate_interest_accrual(const bg::date& start_date,
                                    const bg::date& end_date,
                                    double principal,
                                    const std::string& curve_name);
    
    bg::date find_month_end(const bg::date& date, HolidayCalendar cal);
    
    std::vector<bg::date> generate_payment_schedule(
        const bg::date& start_date,
        const bg::date& end_date,
        int payment_frequency_months,
        HolidayCalendar cal
    );
};

// Usage example:
int main() {
    FinancialCalendar fin_cal;
    
    // Load US federal holidays for 2023
    std::vector<bg::date> us_holidays = {
        bg::date(2023, 1, 1),   // New Year's Day
        bg::date(2023, 1, 16),  // MLK Day
        bg::date(2023, 2, 20),  // Presidents Day
        bg::date(2023, 5, 29),  // Memorial Day
        bg::date(2023, 7, 4),   // Independence Day
        bg::date(2023, 9, 4),   // Labor Day
        bg::date(2023, 10, 9),  // Columbus Day
        bg::date(2023, 11, 11), // Veterans Day
        bg::date(2023, 11, 23), // Thanksgiving
        bg::date(2023, 12, 25)  // Christmas
    };
    
    fin_cal.load_holiday_calendar(FinancialCalendar::HolidayCalendar::US_FEDERAL, 
                                 us_holidays);
    
    // Calculate settlement dates
    bg::date trade_date(2023, 6, 21);
    bg::date settlement = fin_cal.get_settlement_date(
        trade_date, 2, FinancialCalendar::HolidayCalendar::US_FEDERAL
    );
    
    std::cout << "Trade date: " << trade_date << "\n";
    std::cout << "Settlement date (T+2): " << settlement << "\n";
    
    return 0;
}
```

**Learning Goals**:
- Implement complex business day calculations
- Handle multiple holiday calendars
- Calculate financial date sequences
- Understand settlement conventions

### Exercise 3: High-Performance Timing System
**Objective**: Create a hierarchical performance profiler with statistical analysis.

**Implementation Challenge**:
```cpp
class HierarchicalProfiler {
private:
    struct TimingNode {
        std::string name;
        bc::high_resolution_clock::time_point start_time;
        bc::high_resolution_clock::time_point end_time;
        std::vector<std::unique_ptr<TimingNode>> children;
        TimingNode* parent;
        
        bc::duration<double> elapsed() const {
            return bc::duration_cast<bc::duration<double>>(end_time - start_time);
        }
    };
    
    std::unique_ptr<TimingNode> root_;
    TimingNode* current_node_;
    std::map<std::string, std::vector<double>> timing_history_;
    
public:
    // TODO: Implement these methods
    void start_timing(const std::string& name);
    void end_timing();
    void reset();
    
    struct Statistics {
        double mean;
        double median;
        double std_dev;
        double min_time;
        double max_time;
        size_t sample_count;
    };
    
    Statistics get_statistics(const std::string& name) const;
    void print_timing_tree() const;
    void print_summary_report() const;
    
    // RAII helper for automatic timing
    class ScopedTimer {
        HierarchicalProfiler& profiler_;
        public:
            ScopedTimer(HierarchicalProfiler& prof, const std::string& name) 
                : profiler_(prof) {
                profiler_.start_timing(name);
            }
            ~ScopedTimer() {
                profiler_.end_timing();
            }
    };
    
    ScopedTimer scoped_timer(const std::string& name) {
        return ScopedTimer(*this, name);
    }
};

// Macro for easy profiling
#define PROFILE_SCOPE(profiler, name) \
    auto timer = profiler.scoped_timer(name)

// Usage example:
void complex_algorithm(HierarchicalProfiler& profiler) {
    PROFILE_SCOPE(profiler, "ComplexAlgorithm");
    
    {
        PROFILE_SCOPE(profiler, "DataPreprocessing");
        // Simulate preprocessing
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    {
        PROFILE_SCOPE(profiler, "MainComputation");
        
        {
            PROFILE_SCOPE(profiler, "PhaseOne");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        {
            PROFILE_SCOPE(profiler, "PhaseTwo");
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }
    
    {
        PROFILE_SCOPE(profiler, "Postprocessing");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

int main() {
    HierarchicalProfiler profiler;
    
    // Run algorithm multiple times for statistics
    for (int i = 0; i < 10; ++i) {
        complex_algorithm(profiler);
    }
    
    profiler.print_timing_tree();
    profiler.print_summary_report();
    
    return 0;
}
```

**Learning Goals**:
- Master high-resolution timing techniques
- Implement hierarchical data structures
- Calculate statistical measures
- Create RAII timing helpers

### Exercise 4: Calendar Application Backend
**Objective**: Implement a complete calendar system with recurring events and timezone support.

**Implementation Challenge**:
```cpp
class CalendarSystem {
public:
    enum class RecurrenceType {
        DAILY,
        WEEKLY,
        MONTHLY,
        YEARLY,
        CUSTOM
    };
    
    struct RecurrenceRule {
        RecurrenceType type;
        int interval;  // Every N days/weeks/months/years
        std::set<bg::greg_weekday> weekdays;  // For weekly recurrence
        int day_of_month;  // For monthly recurrence
        bg::date until_date;  // End date for recurrence
        int count;  // Number of occurrences (-1 for unlimited)
    };
    
    struct Event {
        std::string id;
        std::string title;
        std::string description;
        bl::local_date_time start_time;
        bp::time_duration duration;
        std::string time_zone;
        std::optional<RecurrenceRule> recurrence;
        std::set<std::string> attendees;
        std::map<std::string, std::string> metadata;
    };
    
private:
    std::map<std::string, Event> events_;
    std::map<std::string, bl::time_zone_ptr> time_zones_;
    
public:
    // TODO: Implement these methods
    std::string create_event(const Event& event);
    bool update_event(const std::string& id, const Event& event);
    bool delete_event(const std::string& id);
    
    std::vector<Event> get_events_in_range(
        const bp::time_period& range,
        const std::string& time_zone
    );
    
    std::vector<bl::local_date_time> calculate_occurrences(
        const Event& event,
        const bp::time_period& range
    );
    
    std::vector<Event> find_conflicts(const Event& new_event);
    
    void generate_calendar_view(
        const bg::date& month,
        const std::string& time_zone
    );
    
    std::vector<Event> search_events(
        const std::string& query,
        const bp::time_period& range = bp::time_period()
    );
};

// Usage example:
int main() {
    CalendarSystem calendar;
    
    // Create a recurring weekly meeting
    CalendarSystem::Event weekly_meeting;
    weekly_meeting.title = "Team Standup";
    weekly_meeting.description = "Weekly team synchronization";
    weekly_meeting.start_time = bl::local_date_time(
        bg::date(2023, 6, 21), 
        bp::time_duration(9, 0, 0),
        bl::time_zone_ptr(new bl::posix_time_zone("EST-5EDT,M4.1.0,M10.5.0")),
        false
    );
    weekly_meeting.duration = bp::hours(1);
    weekly_meeting.time_zone = "EST";
    
    CalendarSystem::RecurrenceRule weekly_rule;
    weekly_rule.type = CalendarSystem::RecurrenceType::WEEKLY;
    weekly_rule.interval = 1;
    weekly_rule.weekdays = {bg::Monday, bg::Wednesday, bg::Friday};
    weekly_rule.until_date = bg::date(2023, 12, 31);
    weekly_meeting.recurrence = weekly_rule;
    
    std::string meeting_id = calendar.create_event(weekly_meeting);
    std::cout << "Created recurring meeting: " << meeting_id << "\n";
    
    return 0;
}
```

**Learning Goals**:
- Implement complex recurring event algorithms
- Handle timezone conversions for global users
- Create efficient calendar data structures
- Implement conflict detection algorithms

## Performance Considerations and Optimization

### Date/Time Operations Performance

**Memory Usage Patterns**:
```cpp
void demonstrate_memory_efficiency() {
    std::cout << "=== Memory Usage Analysis ===\n";
    
    // Size comparison of different types
    std::cout << "Type sizes:\n";
    std::cout << "  bg::date: " << sizeof(bg::date) << " bytes\n";
    std::cout << "  bp::time_duration: " << sizeof(bp::time_duration) << " bytes\n";
    std::cout << "  bp::ptime: " << sizeof(bp::ptime) << " bytes\n";
    std::cout << "  bl::local_date_time: " << sizeof(bl::local_date_time) << " bytes\n";
    std::cout << "  std::time_t: " << sizeof(std::time_t) << " bytes\n";
    
    // Demonstrate efficient vs inefficient patterns
    auto start = bc::high_resolution_clock::now();
    
    // INEFFICIENT: Creating many date objects
    std::vector<std::string> date_strings;
    for (int i = 0; i < 10000; ++i) {
        bg::date d(2023, 1, 1);
        d += bg::days(i);
        date_strings.push_back(to_iso_string(d));
    }
    
    auto inefficient_time = bc::high_resolution_clock::now() - start;
    
    start = bc::high_resolution_clock::now();
    
    // EFFICIENT: Reuse base date and increment
    bg::date base_date(2023, 1, 1);
    date_strings.clear();
    date_strings.reserve(10000);
    
    for (int i = 0; i < 10000; ++i) {
        bg::date d = base_date + bg::days(i);
        date_strings.push_back(to_iso_string(d));
    }
    
    auto efficient_time = bc::high_resolution_clock::now() - start;
    
    std::cout << "\nPerformance comparison (10,000 dates):\n";
    std::cout << "  Inefficient: " 
              << bc::duration_cast<bc::microseconds>(inefficient_time).count() << " μs\n";
    std::cout << "  Efficient: " 
              << bc::duration_cast<bc::microseconds>(efficient_time).count() << " μs\n";
    std::cout << "  Improvement: " 
              << (double)inefficient_time.count() / efficient_time.count() << "x\n";
}
```

**Time Zone Object Management**:
```cpp
class TimeZoneManager {
private:
    static std::map<std::string, bl::time_zone_ptr> timezone_cache_;
    static std::mutex cache_mutex_;
    
public:
    static bl::time_zone_ptr get_timezone(const std::string& posix_tz) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        auto it = timezone_cache_.find(posix_tz);
        if (it != timezone_cache_.end()) {
            return it->second;  // Return cached timezone
        }
        
        // Create new timezone and cache it
        auto tz = bl::time_zone_ptr(new bl::posix_time_zone(posix_tz));
        timezone_cache_[posix_tz] = tz;
        return tz;
    }
    
    static void clear_cache() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        timezone_cache_.clear();
    }
    
    static size_t cache_size() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return timezone_cache_.size();
    }
};

// Usage for better performance
void efficient_timezone_usage() {
    // INEFFICIENT: Creating timezone objects repeatedly
    for (int i = 0; i < 1000; ++i) {
        bl::time_zone_ptr tz(new bl::posix_time_zone("EST-5EDT,M4.1.0,M10.5.0"));
        bl::local_date_time ldt(bg::day_clock::local_day(), bp::hours(12), tz, false);
        // Use ldt...
    }
    
    // EFFICIENT: Reuse cached timezone
    auto tz = TimeZoneManager::get_timezone("EST-5EDT,M4.1.0,M10.5.0");
    for (int i = 0; i < 1000; ++i) {
        bl::local_date_time ldt(bg::day_clock::local_day(), bp::hours(12), tz, false);
        // Use ldt...
    }
}
```

**High-Resolution Timing Best Practices**:
```cpp
class PrecisionTimer {
private:
    bc::steady_clock::time_point start_time_;
    
public:
    void start() {
        // Use steady_clock for duration measurements (not affected by system clock changes)
        start_time_ = bc::steady_clock::now();
    }
    
    template<typename Duration = bc::microseconds>
    typename Duration::rep elapsed() const {
        auto end_time = bc::steady_clock::now();
        return bc::duration_cast<Duration>(end_time - start_time_).count();
    }
    
    // For statistical analysis
    template<typename Func>
    std::vector<double> benchmark(Func&& func, int iterations = 100) {
        std::vector<double> results;
        results.reserve(iterations);
        
        for (int i = 0; i < iterations; ++i) {
            start();
            func();
            results.push_back(elapsed<bc::duration<double, std::milli>>());
        }
        
        return results;
    }
};

void demonstrate_timing_precision() {
    PrecisionTimer timer;
    
    // Test different precision levels
    auto test_function = []() {
        volatile int sum = 0;
        for (int i = 0; i < 1000; ++i) {
            sum += i;
        }
    };
    
    auto measurements = timer.benchmark(test_function, 1000);
    
    // Calculate statistics
    double mean = std::accumulate(measurements.begin(), measurements.end(), 0.0) / measurements.size();
    std::sort(measurements.begin(), measurements.end());
    double median = measurements[measurements.size() / 2];
    double min_time = measurements.front();
    double max_time = measurements.back();
    
    std::cout << "Timing statistics (ms):\n";
    std::cout << "  Mean: " << mean << "\n";
    std::cout << "  Median: " << median << "\n";
    std::cout << "  Min: " << min_time << "\n";
    std::cout << "  Max: " << max_time << "\n";
    std::cout << "  Range: " << max_time - min_time << "\n";
}
```

## Best Practices and Design Patterns

### 1. **Defensive Date Programming**
Always validate date inputs and handle edge cases gracefully.

```cpp
class SafeDateCalculator {
public:
    static std::optional<bg::date> safe_add_months(const bg::date& date, int months) {
        try {
            bg::date result = date + bg::months(months);
            
            // Verify the result makes sense
            if (result.is_not_a_date()) {
                return std::nullopt;
            }
            
            return result;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }
    
    static bool is_valid_business_date(const bg::date& date) {
        // Check if it's a valid date
        if (date.is_not_a_date() || date.is_infinity()) {
            return false;
        }
        
        // Check if it's not a weekend
        if (date.day_of_week() == bg::Saturday || 
            date.day_of_week() == bg::Sunday) {
            return false;
        }
        
        // Additional business rules can be added here
        return true;
    }
};
```

### 2. **Time Zone Handling Strategy**
Always be explicit about time zones and use UTC for storage.

```cpp
class TimeZoneAwareSystem {
private:
    bp::ptime utc_time_;  // Always store in UTC
    
public:
    void set_time(const bl::local_date_time& local_time) {
        utc_time_ = local_time.utc_time();  // Convert to UTC for storage
    }
    
    bl::local_date_time get_local_time(bl::time_zone_ptr tz) const {
        return bl::local_date_time(utc_time_, tz);  // Convert to local for display
    }
    
    std::string format_for_timezone(const std::string& tz_name) const {
        auto tz = TimeZoneManager::get_timezone(tz_name);
        auto local_time = get_local_time(tz);
        return to_simple_string(local_time);
    }
};
```

### 3. **Performance-Critical Date Operations**
Use appropriate precision and avoid unnecessary conversions.

```cpp
class OptimizedDateRange {
private:
    bg::date start_;
    bg::date end_;
    mutable std::optional<int> cached_day_count_;
    
public:
    OptimizedDateRange(const bg::date& start, const bg::date& end) 
        : start_(start), end_(end) {}
    
    int day_count() const {
        if (!cached_day_count_) {
            cached_day_count_ = (end_ - start_).days();
        }
        return *cached_day_count_;
    }
    
    bool contains(const bg::date& date) const {
        // Fast comparison without creating duration objects
        return date >= start_ && date <= end_;
    }
};
```

## Common Pitfalls and How to Avoid Them

### 1. **Month Arithmetic Edge Cases**
```cpp
void demonstrate_month_pitfalls() {
    std::cout << "=== Month Arithmetic Pitfalls ===\n";
    
    // PITFALL: Expecting Jan 31 + 1 month = Feb 31
    bg::date jan31(2023, 1, 31);
    bg::date feb_result = jan31 + bg::months(1);
    
    std::cout << "Jan 31 + 1 month = " << feb_result << " (Feb 28, not Feb 31!)\n";
    
    // SOLUTION: Be explicit about intent
    bg::date feb_last_day = bg::date(2023, 2, 28);  // Or use end_of_month()
    bg::date jan_last_day = bg::date(2023, 1, 31);
    
    if (jan_last_day.day() == bg::gregorian_calendar::end_of_month_day(
            jan_last_day.year(), jan_last_day.month())) {
        // It was the last day of the month, so make next month's last day
        feb_last_day = bg::date(2023, 2, 1).end_of_month();
    }
    
    std::cout << "Intended result: " << feb_last_day << "\n";
}
```

### 2. **Time Zone Conversion Errors**
```cpp
void demonstrate_timezone_pitfalls() {
    std::cout << "=== Time Zone Pitfalls ===\n";
    
    // PITFALL: Assuming time zone offsets are constant
    auto ny_tz = TimeZoneManager::get_timezone("EST-5EDT,M4.1.0,M10.5.0");
    
    // Winter time (EST)
    bl::local_date_time winter_time(
        bg::date(2023, 1, 15), bp::time_duration(12, 0, 0), ny_tz, false
    );
    
    // Summer time (EDT) 
    bl::local_date_time summer_time(
        bg::date(2023, 7, 15), bp::time_duration(12, 0, 0), ny_tz, false
    );
    
    std::cout << "Winter time: " << winter_time << " (UTC: " << winter_time.utc_time() << ")\n";
    std::cout << "Summer time: " << summer_time << " (UTC: " << summer_time.utc_time() << ")\n";
    
    // Notice the UTC times differ by 1 hour due to DST!
    auto utc_diff = summer_time.utc_time() - winter_time.utc_time();
    auto local_diff = summer_time.local_time() - winter_time.local_time();
    
    std::cout << "UTC difference: " << utc_diff << "\n";
    std::cout << "Local difference: " << local_diff << "\n";
}
```

### 3. **Precision Loss in Conversions**
```cpp
void demonstrate_precision_pitfalls() {
    std::cout << "=== Precision Pitfalls ===\n";
    
    // PITFALL: Losing precision in conversions
    bp::time_duration precise_time(14, 30, 15, 123456);  // Microseconds
    
    std::cout << "Original: " << precise_time << "\n";
    std::cout << "Microseconds: " << precise_time.total_microseconds() << "\n";
    
    // Converting through different representations
    long total_seconds = precise_time.total_seconds();
    bp::time_duration from_seconds = bp::seconds(total_seconds);
    
    std::cout << "Through seconds: " << from_seconds << " (precision lost!)\n";
    
    // SOLUTION: Use appropriate precision throughout
    long total_microseconds = precise_time.total_microseconds();
    bp::time_duration restored = bp::microseconds(total_microseconds);
    
    std::cout << "Properly restored: " << restored << "\n";
}
```

## Assessment and Learning Objectives

### Self-Assessment Checklist

By the end of this section, you should be able to:

**Core Concepts:**
□ Explain the difference between dates, times, and time points  
□ Understand time zone representation and conversion  
□ Handle daylight saving time transitions correctly  
□ Implement calendar arithmetic with edge case handling  

**Practical Skills:**
□ Parse and format dates/times in multiple formats  
□ Implement business day calculations with holiday support  
□ Create high-precision timing and profiling systems  
□ Handle recurring events and complex scheduling  

**Advanced Topics:**
□ Optimize date/time operations for performance  
□ Design time zone-aware systems  
□ Implement statistical analysis of timing data  
□ Handle cross-platform date/time differences  

### Practical Assessment Questions

**Conceptual Questions:**
1. Why does January 31 + 1 month = February 28 (or 29)?
2. How do you safely handle time zone conversions during DST transitions?
3. What's the difference between `system_clock` and `steady_clock` for timing?
4. How would you implement a "smart" month addition that preserves end-of-month semantics?

**Implementation Challenges:**
1. Create a function that finds the Nth business day of each month in a year
2. Implement a meeting scheduler that avoids conflicts across time zones
3. Build a performance profiler that can measure nanosecond-level operations
4. Design a holiday calendar system that supports multiple countries

**Debugging Scenarios:**
1. A financial application calculates wrong settlement dates around holidays
2. A global application shows wrong times after daylight saving transitions
3. A performance profiler gives inconsistent results
4. A calendar application creates events at wrong times for international users

## Next Steps

Move on to [File System Operations](06_File_System_Operations.md) to explore Boost's file system capabilities.
