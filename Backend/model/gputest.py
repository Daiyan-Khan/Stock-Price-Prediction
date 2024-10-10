import cudf
import cupy as cp

def test_cudpy():
    # Create a simple DataFrame using cuDF
    df = cudf.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [10, 20, 30, 40, 50]
    })

    print("Original DataFrame:")
    print(df)

    # Perform a simple operation: add a new column
    df['c'] = df['a'] + df['b']
    print("\nDataFrame after adding column 'c':")
    print(df)

    # Convert the DataFrame to a NumPy array using CuPy
    numpy_array = cp.asarray(df['c'].to_array())
    print("\nConverted NumPy array from column 'c':")
    print(numpy_array)

    # Check if the array has been created correctly
    assert numpy_array.shape == (5,), "Array shape mismatch!"

    print("\nTest passed: cudPy is working correctly!")

if __name__ == "__main__":
    test_cudpy()
