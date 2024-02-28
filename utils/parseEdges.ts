import Papa from "papaparse";
import { useEffect, useState } from "react";

type Data = {
  name: string;
  family: string;
  email: string;
  date: string;
  job: string;
};

type Values = {
  data: Data[];
};

const parseEdges = () => {
  const [values, setValues] = useState<Values | undefined>();

  const getCSV = () => {
    Papa.parse("../data/in_g_edge_list1000.csv", {
      header: true,
      download: true,
      skipEmptyLines: true,
      delimiter: ",",
      complete: (results) => {
        if (Array.isArray(results.data)) {
          // Type assertion here to inform TypeScript about the correct type
          setValues({ data: results.data as Data[] });
        } else {
          console.error("No data found in CSV parsing results");
        }
      },
      error: (error) => {
        console.error("Error parsing CSV:", error);
      },
    });
  };

  useEffect(() => {
    getCSV();
  }, []);

  return values;
};

export default parseEdges;
