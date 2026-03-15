---
trigger: model_decision
description: when using cli
---

When fixing segfaults, ensure the loaded model used for testing is loaded not on the main GPU but on a secondary GPU. Since the main GPU is used for the display, it should not be overloaded with heavy computations as it can cause the system to become unresponsive.