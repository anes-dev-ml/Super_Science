#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>

class Matrix
{
private:
    std::vector<float> data;
    int rows;
    int columns;

    void validateCoordinates(int row, int column) const
    {
        if (row < 0 || row >= rows ||
            column < 0 || column >= columns)
        {
            throw std::out_of_range("Matrix coordinates are out of bounds.");
        }
    }

public:
    Matrix(int r, int c) : rows(r), columns(c)
    {
        if (r <= 0 || c <= 0)
        {
            throw std::invalid_argument("Matrix dimensions must be positive.");
        }

        std::size_t safeRows = static_cast<std::size_t>(r);
        std::size_t safeColumns = static_cast<std::size_t>(c);

        if (safeRows > data.max_size() / safeColumns)
        {
            throw std::length_error("Matrix dimensions are too large.");
        }

        data.resize(safeRows * safeColumns, 0.0f);
    }

    int getRows() const
    {
        return rows;
    }

    int getColumns() const
    {
        return columns;
    }

    float &at(int row, int column)
    {
        validateCoordinates(row, column);

        std::size_t index =
            static_cast<std::size_t>(row) *
                static_cast<std::size_t>(columns) +
            static_cast<std::size_t>(column);

        return data[index];
    }

    const float &at(int row, int column) const
    {
        validateCoordinates(row, column);

        std::size_t index =
            static_cast<std::size_t>(row) *
                static_cast<std::size_t>(columns) +
            static_cast<std::size_t>(column);

        return data[index];
    }

    Matrix add(const Matrix &other) const
    {
        if (rows != other.rows || columns != other.columns)
        {
            throw std::invalid_argument("Matrix addition: dimensions must match.");
        }

        Matrix result(rows, columns);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < columns; ++j)
            {
                result.at(i, j) = at(i, j) + other.at(i, j);
            }
        }

        return result;
    }

    Matrix multiplyScalar(float scalar) const
    {
        Matrix result(rows, columns);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < columns; ++j)
            {
                result.at(i, j) = at(i, j) * scalar;
            }
        }

        return result;
    }

    Matrix multiply(const Matrix &other) const
    {
        if (columns != other.rows)
        {
            throw std::invalid_argument("Matrix multiplication: incompatible dimensions.");
        }

        Matrix result(rows, other.columns);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < other.columns; ++j)
            {
                float sum = 0.0f;

                for (int k = 0; k < columns; ++k)
                {
                    sum += at(i, k) * other.at(k, j);
                }

                result.at(i, j) = sum;
            }
        }

        return result;
    }

    Matrix transpose() const
    {
        Matrix result(columns, rows);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < columns; ++j)
            {
                result.at(j, i) = at(i, j);
            }
        }

        return result;
    }
};
