/* Count CPython object-allocator calls without entering Python from the hook.

   Compiled by measure.py against the running interpreter's libpython.
   The hook only increments a counter and forwards to the previous allocator.
*/
#include <stddef.h>
#include <stdint.h>

typedef struct {
    void *ctx;
    void *(*malloc)(void *ctx, size_t size);
    void *(*calloc)(void *ctx, size_t nelem, size_t elsize);
    void *(*realloc)(void *ctx, void *ptr, size_t new_size);
    void (*free)(void *ctx, void *ptr);
} PyMemAllocatorEx;

/* PyMemAllocatorDomain: RAW=0, MEM=1, OBJ=2. */
#define PYMEM_DOMAIN_OBJ 2

void PyMem_GetAllocator(int domain, PyMemAllocatorEx *allocator);
void PyMem_SetAllocator(int domain, PyMemAllocatorEx *allocator);

static uint64_t alloc_events = 0;
static PyMemAllocatorEx original;
static int installed = 0;

static void *hook_malloc(void *ctx, size_t size) {
    alloc_events++;
    return original.malloc(original.ctx, size);
}

static void *hook_calloc(void *ctx, size_t nelem, size_t elsize) {
    alloc_events++;
    return original.calloc(original.ctx, nelem, elsize);
}

static void *hook_realloc(void *ctx, void *ptr, size_t new_size) {
    alloc_events++;
    return original.realloc(original.ctx, ptr, new_size);
}

static void hook_free(void *ctx, void *ptr) {
    original.free(original.ctx, ptr);
}

void alloc_count_install(void) {
    if (!installed) {
        PyMem_GetAllocator(PYMEM_DOMAIN_OBJ, &original);
        PyMemAllocatorEx hooked;
        hooked.ctx = NULL;
        hooked.malloc = hook_malloc;
        hooked.calloc = hook_calloc;
        hooked.realloc = hook_realloc;
        hooked.free = hook_free;
        PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &hooked);
        installed = 1;
    }
    alloc_events = 0;
}

uint64_t alloc_count_read(void) {
    return alloc_events;
}

uint64_t alloc_count_uninstall(void) {
    if (installed) {
        PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &original);
        installed = 0;
    }
    return alloc_events;
}
