int htoi(char s[]) {
    int i = 0, result = 0, value;
    
    // Iterate over the string until null terminator
    while (s[i] != '\0') {
        if (s[i] >= '0' && s[i] <= '9') {  // Handle '0' to '9'
            value = s[i] - '0';
        } else if (s[i] >= 'a' && s[i] <= 'f') {  // Handle 'a' to 'f'
            value = s[i] - 'a' + 10;
        } else if (s[i] >= 'A' && s[i] <= 'F') {  // Handle 'A' to 'F'
            value = s[i] - 'A' + 10;
        } else {  // Ignore invalid characters
            break;
        }
        result = result * 16 + value;  // Update result
        i++;
    }
    return result;
}
