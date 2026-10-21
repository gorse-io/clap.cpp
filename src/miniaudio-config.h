#pragma once

// Decode/convert only: no devices, background workers, engine or encoders.
// Keep these definitions identical in the C implementation and C++ consumers.
#define MA_NO_DEVICE_IO
#define MA_NO_ENGINE
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_THREADING
#define MA_NO_RUNTIME_LINKING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#include "miniaudio.h"
