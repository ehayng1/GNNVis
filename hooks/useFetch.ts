import Papa from "papaparse";

type Callback = (data: any) => void;

const useFetch = () => {
  const sanitizeColumns = (data: any) => {
    return data.map((item: any) => {
      const sanitizedItem: any = {};
      Object.keys(item).forEach((key) => {
        const sanitizedKey = key.toLowerCase().replace(/(\s|-)+/g, "_");
        sanitizedItem[sanitizedKey] = item[key];
      });
      return sanitizedItem;
    });
  };

  const fetchCsvData = async (filePath: string, callback: Callback) => {
    const response = await fetch(filePath);
    const reader = response.body!.getReader();
    const result = await reader.read();
    const decoder = new TextDecoder("utf-8");
    const csvString = decoder.decode(result.value!);
    const { data } = Papa.parse(csvString, {
      header: true,
      dynamicTyping: true,
    });
    console.log("Data: ", data);
    const sanitizedData = sanitizeColumns(data);
    callback(sanitizedData);
  };

  return { fetchCsvData };
};

export default useFetch;
