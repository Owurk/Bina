#include "Event.hpp"

#include <atomic>
#include <chrono>
#include <sstream>

namespace
{
    std::atomic<unsigned long long> eventCounter{ 0 };

    std::string escapeJson(const std::string& value)
    {
        std::ostringstream escaped;

        for (char c : value)
        {
            switch (c)
            {
            case '"':  escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b";  break;
            case '\f': escaped << "\\f";  break;
            case '\n': escaped << "\\n";  break;
            case '\r': escaped << "\\r";  break;
            case '\t': escaped << "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20)
                {
                    const char* hex = "0123456789abcdef";
                    unsigned char byte =
                        static_cast<unsigned char>(c);

                    escaped << "\\u00"
                        << hex[(byte >> 4) & 0x0F]
                        << hex[byte & 0x0F];
                }
                else
                {
                    escaped << c;
                }
            }
        }

        return escaped.str();
    }
}

Event::Event(
    const std::string& snapshotRef,
    const std::string& clipRef,
    const std::string& detectedClass,
    const std::string& description
)
    : timestamp(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    ),
    snapshotRef(snapshotRef),
    clipRef(clipRef),
    detectedClass(detectedClass),
    description(description)
{
    id = "event_" + std::to_string(timestamp) + "_" +
        std::to_string(eventCounter.fetch_add(1));
}

const std::string& Event::getId() const
{
    return id;
}

std::int64_t Event::getTimestamp() const
{
    return timestamp;
}

const std::string& Event::getSnapshotRef() const
{
    return snapshotRef;
}

const std::string& Event::getClipRef() const
{
    return clipRef;
}

const std::string& Event::getDetectedClass() const
{
    return detectedClass;
}

const std::string& Event::getDescription() const
{
    return description;
}

std::string Event::toJson() const
{
    std::ostringstream json;

    json << "{"
        << "\"id\":\"" << escapeJson(id) << "\","
        << "\"timestamp\":" << timestamp << ","
        << "\"snapshotRef\":\"" << escapeJson(snapshotRef) << "\","
        << "\"clipRef\":\"" << escapeJson(clipRef) << "\","
        << "\"detectedClass\":\"" << escapeJson(detectedClass) << "\","
        << "\"description\":\"" << escapeJson(description) << "\""
        << "}";

    return json.str();
}
