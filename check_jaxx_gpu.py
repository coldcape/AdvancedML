#!/usr/bin/env python

import jax

def main():
    # List all devices recognized by JAX
    devices = jax.devices()
    print("Devices recognized by JAX:")
    for device in devices:
        print(f" - {device} (platform: {device.platform})")

    # Check if any device is a GPU
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if gpu_devices:
        print("\nGPU device(s) available:")
        for gpu in gpu_devices:
            print(f" - {gpu}")
    else:
        print("\nNo GPU device available for JAX; it will use the CPU.")

if __name__ == "__main__":
    main()
