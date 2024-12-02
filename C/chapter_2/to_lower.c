int lower(c) /* convert c to lower case; ASCII only */
int c;
{
  if (c >= 'A' && c <= 'Z') {
    return(c + 32);
  }
  return c;
}