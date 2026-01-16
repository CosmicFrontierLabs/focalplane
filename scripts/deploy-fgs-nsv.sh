#!/bin/bash
# Wrapper for deploy-fgs.sh - deploys to NSV (NSV455 camera on orin-416)
# See deploy-fgs.sh for full documentation
exec "$(dirname "$0")/deploy-fgs.sh" nsv "$@"
