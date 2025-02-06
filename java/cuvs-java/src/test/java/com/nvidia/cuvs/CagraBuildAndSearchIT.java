/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;

public class CagraBuildAndSearchIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
  }

  // Sample data and query
  static final float[][] datasetA = {
    { 0.74021935f, 0.9209938f },
    { 0.03902049f, 0.9689629f },
    { 0.92514056f, 0.4463501f },
    { 0.6673192f, 0.10993068f }
  };
  static final List<Integer> mapA = List.of(0, 1, 2, 3);
  static final float[][] queriesA = {
    { 0.48216683f, 0.0428398f },
    { 0.5084142f, 0.6545497f },
    { 0.51260436f, 0.2643005f },
    { 0.05198065f, 0.5789965f }
  };

  // Expected search results
  static final List<Map<Integer, Float>> expectedResultsA = Arrays.asList(
    Map.of(3, 0.038782578f, 2, 0.3590463f, 0, 0.83774555f),
    Map.of(0, 0.12472608f, 2, 0.21700792f, 1, 0.31918612f),
    Map.of(3, 0.047766715f, 2, 0.20332818f, 0, 0.48305473f),
    Map.of(1, 0.15224178f, 0, 0.59063464f, 3, 0.5986642f));

  /**
   * A basic test that checks the whole flow - from indexing to search.
   */
  @Test
  public void testIndexingAndSearchingFlow() throws Throwable {
    for (int j = 0; j < 10; j++) {
      try (CuVSResources resources = CuVSResources.create()) {
        var task = indexAndQueryOnce(datasetA, mapA, queriesA, expectedResultsA, resources);
        task.call();
      }
    }
  }

  @Test
  public void testIndexingAndSearchingFlowThreaded() throws Throwable {
    for (int j = 0; j < 10; j++) {
      try (CuVSResources resources = CuVSResources.create()) {
        var task = indexAndQueryOnce(datasetA, mapA, queriesA, expectedResultsA, resources);

        boolean executeInDifferentThreads = true;
        if (executeInDifferentThreads) {
          var executor = Executors.newSingleThreadExecutor();
          var future = executor.submit(task);
          future.get();
          executor.shutdown();
          boolean terminated = executor.awaitTermination(30, TimeUnit.SECONDS);
          assertTrue(terminated);
        } else {
          task.call();
        }
      }
    }
  }

  static Callable<Void> indexAndQueryOnce(
      float[][] dataset,
      List<Integer> map,
      float[][] queries,
      List<Map<Integer, Float>> expectedResults,
      CuVSResources resources) {
    return () -> {
      try {
        // Configure index parameters
        CagraIndexParams indexParams = new CagraIndexParams.Builder()
                .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
                .withGraphDegree(1)
                .withIntermediateGraphDegree(2)
                .withNumWriterThreads(32)
                .withMetric(CuvsDistanceType.L2Expanded)
                .build();

        // Create the index with the dataset
        CagraIndex index = CagraIndex.newBuilder(resources)
                .withDataset(dataset)
                .withIndexParams(indexParams)
                .build();

        // Saving the index on to the disk.
        String indexFileName = UUID.randomUUID().toString() + ".cag";
        index.serialize(new FileOutputStream(indexFileName));

        // Loading a CAGRA index from disk.
        File indexFile = new File(indexFileName);
        InputStream inputStream = new FileInputStream(indexFile);
        CagraIndex loadedIndex = CagraIndex.newBuilder(resources)
                .from(inputStream)
                .build();

        // Configure search parameters
        CagraSearchParams searchParams = new CagraSearchParams.Builder(resources)
                .build();

        // Create a query object with the query vectors
        CagraQuery cuvsQuery = new CagraQuery.Builder()
                .withTopK(3)
                .withSearchParams(searchParams)
                .withQueryVectors(queries)
                .withMapping(map)
                .build();

        // Perform the search
        SearchResults results = index.search(cuvsQuery);

        // Check results
        log.info(results.getResults().toString());
        assertEquals(expectedResults, results.getResults());

        // Search from deserialized index
        results = loadedIndex.search(cuvsQuery);

        // Check results
        log.info(results.getResults().toString());
        assertEquals(expectedResults, results.getResults());

        // Cleanup
        if (indexFile.exists()) {
          indexFile.delete();
        }
        index.destroyIndex();
      } catch (Throwable t) {
        handleThrowable(t);
      }
      return null;
    };
  }

//  // This is a non-realistic trivial dataset, with just a single small
//  // vector. It demonstrates that trying to build a cagra index with a
//  // single vector fails. Such a scenario can happen when data is indexed
//  // and merged across shards, e.g. in Lucene or Elasticsearch.
//  //@Test
//  public void testIndexAndSearchSingle() throws Throwable {
//    float[][] dataset = {
//        { 0.74021935f, 0.9209938f }
//    };
//
//    try (CuVSResources resources = CuVSResources.create()) {
//      CagraIndexParams indexParams = new CagraIndexParams.Builder()
//              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
//              .withGraphDegree(1)
//              .withIntermediateGraphDegree(2)
//              .withNumWriterThreads(32)
//              .withMetric(CuvsDistanceType.L2Expanded)
//              .build();
//      CagraIndex index = CagraIndex.newBuilder(resources)
//              .withDataset(dataset)
//              .withIndexParams(indexParams)
//              .build();
//      index.destroyIndex();
//    }
//  }

  static void handleThrowable(Throwable t) throws IOException {
    switch (t) {
      case IOException ioe -> throw ioe;
      case Error error -> throw error;
      case RuntimeException re -> throw re;
      case null, default -> throw new RuntimeException("UNEXPECTED: exception type", t);
    }
  }
}
