#!/bin/sh
set -eu

config_path="${RAGKIT_CONFIG:-/app/config/ragkit.toml}"

if [ ! -r "$config_path" ]; then
    echo "ragkit startup error: configured profile is not readable" >&2
    exit 2
fi
if [ ! -r /data/corpus ] || [ ! -d /data/corpus ]; then
    echo "ragkit startup error: corpus mount is not a readable directory" >&2
    exit 2
fi
if [ ! -w /var/lib/ragkit ]; then
    echo "ragkit startup error: persistent state mount is not writable" >&2
    exit 2
fi
if [ "$#" -eq 0 ]; then
    echo "ragkit startup error: no service command was supplied" >&2
    exit 2
fi

exec "$@"
