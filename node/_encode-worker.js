'use strict';
const { parentPort, workerData } = require('node:worker_threads');
const native = require('./_native');
// Koffi 3 represents pointers as BigInt (official pointers documentation).
// Worker isolates share this process/address space. Only the address travels;
// the main-isolate lease owns it until this worker exits. Never destroy it here.
const { pointer, operation, args } = workerData;
if (typeof pointer !== 'bigint') throw new TypeError('Invalid private CLAP handle');
try {
  const value = native.embedding(native[operation], [pointer, ...args]);
  parentPort.postMessage({ value }, [value.buffer]);
} catch (error) {
  // embedding captured this OS thread's TLS diagnostic before freeing output.
  parentPort.postMessage({ error: { name: error.name, message: error.message } });
}
