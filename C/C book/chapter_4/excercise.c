#include <stdio.h>

int main()
{
    char line[256];
    // Initialize memory to zeros so that it is already zero-terminated
    char memory[256] = {0};
    char opcode;
    int count, address, value;

    while (fgets(line, 256, stdin) != NULL) {
        //printf("\nLine: %s\n", line);

        // 'X' means exit
        if (line[0] == 'X') {
            break;
        }
        // '*' is a comment - just print it and continue
        if (line[0] == '*') {
            printf("%s", line);
            continue;
        }

        // Look for input lines like "5 = 42" or "10 + 3" or "7 - 1"
        count = sscanf(line, "%d %c %d", &address, &opcode, &value);
        if (count != 3) {
            // If it doesn't match the pattern, ignore and loop
            continue;
        }

        // Debug: show what we parsed
        //printf("address: %d opcode: %c value: %d\n", address, opcode, value);

        // Perform the operation on memory[address]
        switch (opcode) {
        case '=':
            // Set memory[address] to value
            memory[address] = (char)value;
            break;
        case '+':
            // Add value to memory[address]
            memory[address] = (char)((unsigned char)memory[address] + value);
            break;
        case '-':
            // Subtract value from memory[address]
            memory[address] = (char)((unsigned char)memory[address] - value);
            break;
        default:
            // Any other opcode is ignored
            break;
        }

    }

    // Final print of memory after we reach 'X' or EOF
    printf("Memory:\n%s\n", memory);
    return 0;
}