#pragma once

#include <cstdint>
#include <string>

class Event
{
public:
    Event(
        const std::string& snapshotRef,
        const std::string& clipRef,
        const std::string& detectedClass,
        const std::string& description = ""
    );

    const std::string& getId() const;
    std::int64_t getTimestamp() const;
    const std::string& getSnapshotRef() const;
    const std::string& getClipRef() const;
    const std::string& getDetectedClass() const;
    const std::string& getDescription() const;

    std::string toJson() const;

private:
    std::string id;
    std::int64_t timestamp;
    std::string snapshotRef;
    std::string clipRef;
    std::string detectedClass;
    std::string description;
};
