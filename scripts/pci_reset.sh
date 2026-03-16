#!/bin/bash

# Script to reset PCI device using flatpak-spawn --host
# Usage: ./pci_reset.sh

echo "1" | flatpak-spawn --host sudo tee /sys/bus/pci/devices/0000:04:00.0/reset
