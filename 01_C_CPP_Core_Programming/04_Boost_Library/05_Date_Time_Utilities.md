# Date and Time Utilities

*Duration: 1 week*

## Overview

This section covers Boost's date and time handling libraries, including calendar operations, time zone support, and high-resolution timing capabilities.

## Learning Topics

### Boost.DateTime
- Date and time representation and manipulation
- Time zone handling and conversions
- Date/time parsing and formatting
- Date/time arithmetic and comparisons
- Calendar calculations and business logic

### Boost.Chrono
- High-resolution timing and duration measurements
- Clock types and time point representations
- Duration arithmetic and conversions
- Performance timing and benchmarking

## Code Examples

### Basic Date Operations
```cpp
#include <boost/date_time/gregorian/gregorian.hpp>
#include <iostream>

namespace bg = boost::gregorian;

void demonstrate_date_basics() {
    // Create dates
    bg::date today = bg::day_clock::local_day();
    bg::date specific_date(2023, 6, 21);
    bg::date from_string = bg::from_string("2023/12/25");
    
    std::cout << "Today: " << today << "\n";
    std::cout << "Specific date: " << specific_date << "\n";
    std::cout << "From string: " << from_string << "\n";
    
    // Date arithmetic
    bg::date_duration one_week(7);
    bg::date next_week = today + one_week;
    bg::date last_month = today - bg::months(1);
    
    std::cout << "Next week: " << next_week << "\n";
    std::cout << "Last month: " << last_month << "\n";
    
    // Date comparisons
    if (today < from_string) {
        bg::date_duration diff = from_string - today;
        std::cout << "Christmas is " << diff.days() << " days away\n";
    }
}

void demonstrate_date_properties() {
    bg::date date(2023, 6, 21);
    
    std::cout << "Date: " << date << "\n";
    std::cout << "Year: " << date.year() << "\n";
    std::cout << "Month: " << date.month() << "\n";
    std::cout << "Day: " << date.day() << "\n";
    std::cout << "Day of week: " << date.day_of_week() << "\n";
    std::cout << "Day of year: " << date.day_of_year() << "\n";
    std::cout << "Week number: " << date.week_number() << "\n";
    
    // Check properties
    std::cout << "Is leap year: " << std::boolalpha 
              << bg::gregorian_calendar::is_leap_year(date.year()) << "\n";
}
```

### Date Ranges and Iterations
```cpp
#include <boost/date_time/gregorian/gregorian.hpp>
#include <iostream>

namespace bg = boost::gregorian;

void demonstrate_date_ranges() {
    bg::date start(2023, 6, 1);
    bg::date end(2023, 6, 30);
    
    // Date period
    bg::date_period june(start, end);
    std::cout << "June period: " << june << "\n";
    std::cout << "Length: " << june.length().days() << " days\n";
    
    // Check if date is in period
    bg::date test_date(2023, 6, 15);
    if (june.contains(test_date)) {
        std::cout << test_date << " is in June\n";
    }
    
    // Intersect periods
    bg::date_period summer(bg::date(2023, 6, 1), bg::date(2023, 8, 31));
    bg::date_period vacation(bg::date(2023, 7, 15), bg::date(2023, 8, 15));
    
    bg::date_period intersection = june.intersection(vacation);
    if (!intersection.is_null()) {
        std::cout << "June-vacation overlap: " << intersection << "\n";
    }
}

void demonstrate_date_iteration() {
    bg::date start(2023, 6, 1);
    bg::date end(2023, 6, 7);
    
    std::cout << "Weekdays in first week of June:\n";
    for (bg::day_iterator it(start); it <= end; ++it) {
        if (it->day_of_week() != bg::Sunday && it->day_of_week() != bg::Saturday) {
            std::cout << "  " << *it << " (" << it->day_of_week() << ")\n";
        }
    }
    
    std::cout << "\nMondays in June 2023:\n";
    bg::date june_start(2023, 6, 1);
    bg::date june_end(2023, 6, 30);
    
    for (bg::week_iterator wit(june_start); wit <= june_end; ++wit) {
        if (wit->day_of_week() == bg::Monday) {
            std::cout << "  " << *wit << "\n";
        }
    }
}
```

### Time and DateTime Operations
```cpp
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>

namespace bp = boost::posix_time;
namespace bg = boost::gregorian;

void demonstrate_time_basics() {
    // Create times
    bp::time_duration morning(8, 30, 0);  // 8:30:00
    bp::time_duration afternoon = bp::hours(14) + bp::minutes(45) + bp::seconds(30);
    
    std::cout << "Morning: " << morning << "\n";
    std::cout << "Afternoon: " << afternoon << "\n";
    
    // Time arithmetic
    bp::time_duration meeting_length = bp::hours(1) + bp::minutes(30);
    bp::time_duration meeting_end = morning + meeting_length;
    
    std::cout << "Meeting ends at: " << meeting_end << "\n";
    
    // Time properties
    std::cout << "Hours: " << afternoon.hours() << "\n";
    std::cout << "Minutes: " << afternoon.minutes() << "\n";
    std::cout << "Total seconds: " << afternoon.total_seconds() << "\n";
}

void demonstrate_datetime_operations() {
    // Create date-time
    bg::date today = bg::day_clock::local_day();
    bp::time_duration now_time = bp::second_clock::local_time().time_of_day();
    bp::ptime now(today, now_time);
    
    std::cout << "Current date-time: " << now << "\n";
    
    // Specific date-time
    bp::ptime meeting(bg::date(2023, 6, 21), bp::time_duration(14, 30, 0));
    std::cout << "Meeting time: " << meeting << "\n";
    
    // DateTime arithmetic
    bp::time_duration duration_until_meeting = meeting - now;
    if (!duration_until_meeting.is_negative()) {
        std::cout << "Time until meeting: " << duration_until_meeting << "\n";
    } else {
        std::cout << "Meeting was " << -duration_until_meeting << " ago\n";
    }
    
    // Add time periods
    bp::ptime one_week_later = now + bg::days(7);
    bp::ptime three_hours_later = now + bp::hours(3);
    
    std::cout << "One week later: " << one_week_later << "\n";
    std::cout << "Three hours later: " << three_hours_later << "\n";
}
```

### Date/Time Parsing and Formatting
```cpp
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <locale>

namespace bg = boost::gregorian;
namespace bp = boost::posix_time;

void demonstrate_parsing_formatting() {
    // Parse dates from strings
    std::vector<std::string> date_strings = {
        "2023-06-21",
        "2023/06/21",
        "21-Jun-2023",
        "June 21, 2023"
    };
    
    std::cout << "Parsing dates:\n";
    for (const auto& date_str : date_strings) {
        try {
            bg::date parsed;
            if (date_str.find('-') != std::string::npos && date_str.length() == 10) {
                parsed = bg::from_string(date_str);
            } else if (date_str.find('/') != std::string::npos) {
                parsed = bg::from_us_string(date_str);
            } else {
                parsed = bg::from_string(date_str);
            }
            std::cout << "  '" << date_str << "' -> " << parsed << "\n";
        } catch (const std::exception& e) {
            std::cout << "  '" << date_str << "' -> Parse error: " << e.what() << "\n";
        }
    }
    
    // Format dates
    bg::date date(2023, 6, 21);
    std::cout << "\nFormatting date " << date << ":\n";
    std::cout << "  ISO format: " << to_iso_string(date) << "\n";
    std::cout << "  ISO extended: " << to_iso_extended_string(date) << "\n";
    std::cout << "  Simple: " << to_simple_string(date) << "\n";
    
    // Parse and format times
    bp::ptime dt = bp::time_from_string("2023-06-21 14:30:15");
    std::cout << "\nParsed datetime: " << dt << "\n";
    std::cout << "ISO format: " << to_iso_string(dt) << "\n";
    std::cout << "ISO extended: " << to_iso_extended_string(dt) << "\n";
}

void demonstrate_custom_formatting() {
    bg::date date(2023, 6, 21);
    
    // Custom date formatting using facets
    std::locale loc(std::cout.getloc(), 
                   new bg::date_facet("%A, %B %d, %Y"));
    std::cout.imbue(loc);
    std::cout << "Custom format: " << date << "\n";
    
    // Reset locale
    std::cout.imbue(std::locale::classic());
    
    // Manual formatting
    std::string months[] = {"", "January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"};
    
    std::cout << "Manual format: " << months[date.month()] << " " 
              << date.day() << ", " << date.year() << "\n";
}
```

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

1. **Event Scheduler**
   - Create a meeting scheduler that handles time zones
   - Calculate optimal meeting times across multiple time zones
   - Handle daylight saving time transitions

2. **Financial Calculator**
   - Implement business day calculations for settlements
   - Calculate interest accrual over date ranges
   - Handle holiday calendars for different countries

3. **Performance Profiler**
   - Create a hierarchical timer system
   - Measure function execution times with high precision
   - Generate performance reports with statistics

4. **Calendar Application**
   - Implement recurring event calculations
   - Support different calendar views (monthly, weekly)
   - Handle time zone conversions for global events

## Performance Considerations

### Date/Time Operations
- Cache parsed date/time objects when possible
- Use appropriate precision for timing measurements
- Consider time zone database update requirements

### Memory Usage
- Time zone objects can be memory-intensive
- Share time zone instances across operations
- Use appropriate duration types for range requirements

## Best Practices

1. **Time Zone Handling**
   - Always specify time zones explicitly
   - Handle daylight saving time transitions
   - Use UTC for storage, local time for display
   - Validate time zone data sources

2. **Date Arithmetic**
   - Be careful with month/year arithmetic
   - Handle leap years and month boundaries correctly
   - Use business day calculations for financial applications

3. **Performance Timing**
   - Use steady clocks for duration measurements
   - Consider clock resolution for precision requirements
   - Implement proper statistical analysis for benchmarks

## Assessment

- Can perform complex date and time calculations
- Understands time zone handling and conversions
- Implements business logic with calendar constraints
- Can measure and analyze performance timing accurately

## Next Steps

Move on to [File System Operations](06_File_System_Operations.md) to explore Boost's file system capabilities.
