#include <stdio.h>

/*
 * This example explains pointers in C - the foundation for understanding CUDA memory.
 * Pointers are variables that store memory addresses.
 */

// ============================================================
// Function examples: Pass by value vs pass by reference
// ============================================================

// Pass by VALUE - gets a COPY of the number
void tryToDouble_byValue(int x) {
    x = x * 2;
    printf("  Inside function: x = %d (modified copy)\n", x);
    // This change is LOCAL - original variable unchanged
}

// Pass by REFERENCE - gets a POINTER to the number
void tryToDouble_byReference(int* x) {
    *x = *x * 2;  // * means "dereference" - access the value at that address
    printf("  Inside function: *x = %d (modified original!)\n", *x);
    // This changes the ORIGINAL variable
}

// Function that needs to "return" multiple values via pointers
void getDimensions(int* width, int* height) {
    *width = 1920;
    *height = 1080;
    // Can't return multiple values in C, so we use pointers to modify caller's variables
}

int main() {
    printf("=== PART 1: Basic Variables and Pointers ===\n\n");

    // ============================================================
    // Declaring different types
    // ============================================================

    // Regular int - stores a number
    int number = 42;
    printf("1. Regular int:\n");
    printf("   int number = 42;\n");
    printf("   Value: %d\n", number);
    printf("   Address: %p (where it lives in memory)\n\n", (void*)&number);

    // Pointer to int - stores an ADDRESS of an int
    int* ptr = &number;  // & means "address of"
    printf("2. Pointer to int:\n");
    printf("   int* ptr = &number;\n");
    printf("   ptr stores address: %p\n", (void*)ptr);
    printf("   *ptr (dereference) = %d (the value at that address)\n\n", *ptr);

    // Pointer to pointer to int - stores ADDRESS of a pointer
    int** ptr_ptr = &ptr;
    printf("3. Pointer to pointer to int:\n");
    printf("   int** ptr_ptr = &ptr;\n");
    printf("   ptr_ptr stores address: %p (address of ptr)\n", (void*)ptr_ptr);
    printf("   *ptr_ptr = %p (dereferences once -> gets ptr)\n", (void*)*ptr_ptr);
    printf("   **ptr_ptr = %d (dereferences twice -> gets number)\n\n", **ptr_ptr);

    printf("=== Memory Layout Visualization ===\n");
    printf("   number:     [  42  ] at address %p\n", (void*)&number);
    printf("   ptr:        [  %p  ] at address %p\n", (void*)ptr, (void*)&ptr);
    printf("   ptr_ptr:    [  %p  ] at address %p\n", (void*)ptr_ptr, (void*)&ptr_ptr);
    printf("                  ↓ points to\n");
    printf("                  ↓ points to number\n\n");

    printf("=== PART 2: The & and * Operators ===\n\n");

    int value = 100;
    printf("4. Understanding & (address-of) and * (dereference):\n");
    printf("   int value = 100;\n");
    printf("   value      = %d      (the value itself)\n", value);
    printf("   &value     = %p   (address of value)\n", (void*)&value);
    printf("\n");

    int* pointer = &value;
    printf("   int* pointer = &value;\n");
    printf("   pointer    = %p   (stores address of value)\n", (void*)pointer);
    printf("   *pointer   = %d      (dereference -> gets value)\n\n", *pointer);

    // Modify through pointer
    *pointer = 200;
    printf("5. Modifying through pointer:\n");
    printf("   *pointer = 200;\n");
    printf("   value is now: %d (changed through pointer!)\n\n", value);

    printf("=== PART 3: Pass by Value vs Pass by Reference ===\n\n");

    int num = 10;
    printf("6. Pass by VALUE (gets a copy):\n");
    printf("   Before: num = %d\n", num);
    tryToDouble_byValue(num);
    printf("   After:  num = %d (UNCHANGED - function got a copy)\n\n", num);

    printf("7. Pass by REFERENCE (gets a pointer):\n");
    printf("   Before: num = %d\n", num);
    tryToDouble_byReference(&num);  // Pass ADDRESS of num
    printf("   After:  num = %d (CHANGED - function modified original!)\n\n", num);

    printf("=== PART 4: Why Pointers Matter for Functions ===\n\n");

    // Can't return multiple values directly in C, so use pointers
    int w, h;
    getDimensions(&w, &h);  // Pass addresses so function can modify them
    printf("8. Getting multiple 'return' values via pointers:\n");
    printf("   getDimensions(&w, &h);\n");
    printf("   width = %d, height = %d\n\n", w, h);

    printf("=== PART 5: Why CUDA Uses Pointers ===\n\n");
    printf("This is why CUDA functions like cudaMalloc() look like:\n");
    printf("   cudaMalloc(&gpu_ptr, size);\n");
    printf("\n");
    printf("NOT:\n");
    printf("   gpu_ptr = cudaMalloc(size);  // ❌ This doesn't work\n");
    printf("\n");
    printf("Because cudaMalloc needs to:\n");
    printf("1. Allocate memory on the GPU\n");
    printf("2. MODIFY YOUR POINTER to point to that GPU memory\n");
    printf("3. Also return an error code (so it can't return the pointer too)\n");
    printf("\n");
    printf("Solution: Pass a POINTER TO YOUR POINTER (&gpu_ptr)\n");
    printf("Then cudaMalloc can modify your pointer to point to GPU memory!\n\n");

    printf("=== Summary ===\n");
    printf("Declarations:\n");
    printf("  int x;          // regular int\n");
    printf("  int* ptr;       // pointer to int (stores address)\n");
    printf("  int** ptr_ptr;  // pointer to pointer to int\n\n");
    printf("Operators:\n");
    printf("  &x              // address of x\n");
    printf("  *ptr            // dereference ptr (get value at address)\n\n");
    printf("Function parameters:\n");
    printf("  func(x)         // pass by value (copy)\n");
    printf("  func(&x)        // pass by reference (address)\n");
    printf("  void func(int* p) { *p = 5; }  // modifies original\n");

    return 0;
}

/*
 * HOW TO COMPILE AND RUN:
 *
 * nvcc 02-pointer-basics.cu -o 02-pointer-basics
 * ./02-pointer-basics
 *
 * EXPECTED OUTPUT:
 * Detailed visualization of pointers, addresses, and values
 * Demonstrations of pass-by-value vs pass-by-reference
 * Explanation of why CUDA uses pointer-to-pointer syntax
 *
 * KEY CONCEPTS:
 * - & (address-of) gets the memory address of a variable
 * - * (dereference) gets the value at a memory address
 * - int* is a pointer that stores an address
 * - int** is a pointer to a pointer
 * - Pass by reference allows functions to modify original variables
 * - CUDA uses pointers because GPU and CPU have separate memory
 */
